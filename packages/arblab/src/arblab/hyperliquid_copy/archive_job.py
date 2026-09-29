"""One bounded chronological acquisition/compaction batch per invocation.

Qualified owned staging is disposed; no research eligibility. A source argument is explicit acquisition
authority at the library boundary; the CLI separately requires operator consent.
"""

from pathlib import Path
import shutil

import lz4
import pyarrow

from . import archive_job_artifacts as artifacts
from . import archive_job_qualification as qualification
from . import archive_job_cleanup as cleanup
from .archive_budget import ArchiveBudget, BudgetedArchiveSource, MAX_JOB_BYTES
from .archive_cache import VerifiedArchiveCache, CachedArchiveSource, _safe
from .archive_job_store import ArchiveJobStore
from .archive_plan import plan_archive
from .compact_catalog import CompactCatalog
from .contracts import symbol, semantic_hash
from .download import file_hash
from .proxy_archive_download import download_archive, resume_archive, MAX_BYTES
from .proxy_archive_import import import_archive
from .proxy_compact import compact_history, _sync
from .proxy_compact import MAX_BYTES as MAX_COMPACT_BATCH_BYTES

MAX_CANONICAL_BYTES = 64 * 1024**3
FREE_BATCH_BYTES = (6 + 8 + 8) * 1024**3 + 64 * 1024**2


def _engine():
    root = Path(__file__).parent
    names = (
        "archive_job.py",
        "archive_job_store.py",
        "archive_job_artifacts.py",
        "archive_job_qualification.py",
        "archive_job_cleanup.py",
        "archive_engine_transition.py",
        "archive_transition_evidence.py",
        "archive_transition_snapshot.py",
        "archive.py",
        "contracts.py",
        "download.py",
        "proxy_archive_import.py",
        "proxy_archive_download.py",
        "proxy_compact.py",
        "daily_compact.py",
        "compact_catalog.py",
        "archive_budget.py",
        "archive_cache.py",
        "archive_plan.py",
    )
    return dict(
        schema=1,
        pyarrow=pyarrow.__version__,
        lz4=lz4.__version__,
        code={name: file_hash(root / name) for name in names},
        qualification=qualification._engine(),
    )


class ArchiveJob:
    @classmethod
    def create(
        cls,
        inventory,
        root,
        coins,
        *,
        max_download_bytes,
        cache_manifests=(),
        max_batch_bytes=MAX_BYTES,
    ):
        if (
            type(max_download_bytes) is not int
            or not 0 <= max_download_bytes <= MAX_JOB_BYTES
        ):
            raise ValueError("Invalid job download budget")
        coins = sorted(symbol(c) for c in coins)
        if not 1 <= len(coins) <= 50 or len(set(coins)) != len(coins):
            raise ValueError("Invalid job ingestion markets")
        plan = plan_archive(inventory, max_batch_bytes=max_batch_bytes)
        if not plan["transfer_ready"]:
            raise ValueError(
                "Unsupported oversized archive objects; no job acquisition started"
            )
        objects = [o for b in plan["batches"] for o in b["objects"]]
        cache = VerifiedArchiveCache(cache_manifests, objects)
        if plan["total_bytes"] - cache.total_bytes > max_download_bytes:
            raise ValueError("Job budget cannot cover planned uncached bytes")
        store = ArchiveJobStore.create(
            root,
            dict(
                plan=plan,
                coins=coins,
                cache=cache.evidence,
                engine=_engine(),
                max_download_bytes=max_download_bytes,
            ),
        )
        if max_download_bytes:
            ArchiveBudget(store.root / "budget.sqlite3", objects, max_download_bytes)
        _sync(store.root)
        return cls(store.root)

    def __init__(self, root):
        self.store = ArchiveJobStore(root)
        self.metadata = self.store.metadata
        self._check_engine(recheck_content=True)
        self.batches = self.metadata["plan"]["batches"]
        self.objects = [o for b in self.batches for o in b["objects"]]
        self.budget = None
        if self.metadata["max_download_bytes"]:
            path = self.store.root / "budget.sqlite3"
            _safe(path)
            if not path.is_file():
                raise ValueError(
                    "Missing lifetime budget; refusing to reset reservations"
                )
            self.budget = ArchiveBudget(
                path, self.objects, self.metadata["max_download_bytes"], must_exist=True
            )

    def _check_engine(self, *, recheck_content=False):
        from .archive_engine_transition import active_transition

        current = _engine()
        self.transition = active_transition(
            self.store, current_engine=current, recheck_content=recheck_content
        )
        if self.transition is None and self.metadata["engine"] != current:
            raise ValueError("Archive job parser/engine changed")

    def _save(self, index, stage, path, progress):
        identity = file_hash(path)
        if progress:
            progress(dict(batch=index, published=stage, manifest=str(path)))
        self.store.record(index, stage, path, identity)

    def _raw(self, index, source, progress):
        batch = self.batches[index]
        path = self.store.find(index, "raw")
        if path is not None:
            artifacts.raw(path, batch)  # Scope before any resume GET.
            path = resume_archive(source, path, progress=progress)
        else:
            path = download_archive(
                source,
                batch["start"],
                batch["end"],
                self.store.stage_root(index, "raw"),
                max_bytes=batch["bytes"],
                progress=progress,
            )
        data = artifacts.raw(path, batch)
        if not data["complete"]:
            raise ValueError("Incomplete raw acquisition")
        self._save(index, "raw", path, progress)

    def _normalized(self, index, progress):
        raw = self.store.records()[index, "raw"]
        path = self.store.find(index, "normalized")
        if path is None:
            path = import_archive(
                raw["path"],
                self.metadata["coins"],
                self.store.stage_root(index, "normalized"),
                retain_boundary_spill=True,
                progress=progress,
            )
        artifacts.normalized(path, self.batches[index], self.metadata["coins"], raw)
        self._save(index, "normalized", path, progress)

    def _compact(self, index, retained_bytes, progress):
        records = self.store.records()
        path = self.store.find(index, "compact")
        if path is None:
            path = compact_history(
                records[index, "normalized"]["path"],
                self.store.stage_root(index, "compact"),
                partitioning="source_day",
            )
        data = artifacts.compact(
            path,
            self.batches[index],
            self.metadata["coins"],
            records[index, "raw"],
            records[index, "normalized"],
        )
        if retained_bytes + data["output_bytes"] > MAX_CANONICAL_BYTES:
            raise ValueError("Job canonical history byte limit exceeded")
        self._catalog(path)
        self._save(index, "compact", path, progress)
        return path

    def _catalog(self, path):
        catalog_path = self.store.root / "catalog.sqlite3"
        _safe(catalog_path)
        _safe(catalog_path.with_name(catalog_path.name + "-journal"))
        CompactCatalog(catalog_path).register(path)
        _sync(catalog_path)
        _sync(self.store.root)

    def run_next(self, *, source=None, progress=None):
        with self.store.locked():
            self._check_engine()
            if self.budget is not None:
                self.budget.reserved_bytes  # Require spending history before even HEAD.
            records, retained, completed = self.store.records(), 0, 0
            for index, batch in enumerate(self.batches):
                if (index, "compact") not in records:
                    break
                path = self.store.find(index, "compact")
                data = artifacts.compact(
                    path,
                    batch,
                    self.metadata["coins"],
                    records[index, "raw"],
                    records[index, "normalized"],
                )
                self._catalog(path)
                retained += data["output_bytes"]
                completed += 1
                if (index, "qualified") not in records:
                    report = qualification.complete(self.store, index, progress)
                    self._save(index, "qualified", report, progress)
                    cleanup.dispose_prefix(self.store, completed, progress)
                    return self._result("prefix_qualified", completed, path)
                report = self.store.find(index, "qualified")
                qualification.verify(
                    self.store,
                    index,
                    qualification.pin(records[index, "qualified"]),
                    recheck_content=False,
                )
            if completed:
                qualification.verify(
                    self.store,
                    completed - 1,
                    qualification.pin(records[completed - 1, "qualified"]),
                )
            if retained > MAX_CANONICAL_BYTES:
                raise ValueError("Job canonical history byte limit exceeded")
            cleanup.dispose_prefix(self.store, completed, progress)
            if completed == len(self.batches):
                return self._result("all_batches_qualified", completed)
            # Reserve the full existing per-batch bound, including a recovered
            # uncommitted compact batch, before acquisition or further writes.
            if retained + MAX_COMPACT_BATCH_BYTES > MAX_CANONICAL_BYTES:
                raise ValueError(
                    "Insufficient remaining canonical capacity for bounded batch"
                )
            if shutil.disk_usage(self.store.root).free < FREE_BATCH_BYTES:
                raise ValueError("Insufficient free disk for bounded archive batch")
            cache = VerifiedArchiveCache(
                [e["manifest"] for e in self.metadata["cache"]], self.objects
            )
            if cache.evidence != self.metadata["cache"]:
                raise ValueError("Frozen archive cache identity changed")
            if source is not None and self.budget is None:
                raise ValueError("Network source forbidden for zero-budget job")
            remote = (
                BudgetedArchiveSource(source, self.budget)
                if source is not None
                else None
            )
            adapter = CachedArchiveSource(cache, remote)
            self._raw(completed, adapter, progress)
            self._normalized(completed, progress)
            path = self._compact(completed, retained, progress)
            report = qualification.complete(self.store, completed, progress)
            self._save(completed, "qualified", report, progress)
            cleanup.dispose_prefix(self.store, completed + 1, progress)
            return self._result("prefix_qualified", completed + 1, path)

    def engine_evidence(self):
        self._check_engine()
        return dict(
            original=semantic_hash(self.metadata["engine"]),
            effective=semantic_hash(
                self.transition["new_engine"]
                if self.transition
                else self.metadata["engine"]
            ),
            transition=self.transition["pin"] if self.transition else None,
            baseline=self.transition["baseline"] if self.transition else None,
            snapshot=self.transition["snapshot"] if self.transition else None,
        )

    def _result(self, phase, completed, path=None):
        return dict(
            phase=phase,
            completed_batches=completed,
            batch_count=len(self.batches),
            reserved_bytes=self.budget.reserved_bytes if self.budget else 0,
            manifest=str(path) if path else None,
            qualification="canonical_prefix_validated",
            qualification_report=qualification.pin(
                self.store.records()[completed - 1, "qualified"]
            ),
            research_eligible=False,
            cleanup=cleanup.summary(self.store),
            archive_engine=self.engine_evidence(),
        )
