"""Single local coordinator with cancellation and fail-closed artifact publication."""

import fcntl
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from threading import Event, RLock, Thread

from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.lab_config_v2 import LabConfigV2
from .lab_datasets import DatasetCatalog
from .lab_store import ExperimentStore


def source_fingerprint():
    import arblab.hyperliquid_copy.lab_pipeline as engine

    folder = Path(engine.__file__).parent
    return {p.name: file_hash(p) for p in sorted(folder.glob("*.py"))}


class LabJobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.store = ExperimentStore(self.root)
        self.catalog = DatasetCatalog(self.root)
        self.lock = RLock()
        self.stop = Event()
        self.process = None
        self.active = None
        self.ownership = None

    def start(self):
        self.ownership = (self.root / "worker.lock").open("a")
        fcntl.flock(self.ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.store.recover()
        self.thread = Thread(
            target=self.loop, name="hyperliquid-lab-worker", daemon=True
        )
        self.thread.start()

    def close(self):
        self.stop.set()
        with self.lock:
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None
                self.store.transition(
                    self.active,
                    "running",
                    "failed",
                    error="Interrupted by server shutdown",
                )
                self.active = None
        self.thread.join(timeout=6)
        if self.ownership:
            self.ownership.close()

    def submit(self, request, *, kind="backtest"):
        config = request.config
        provenance = self.catalog.preflight(request.dataset_id, config)
        if request.parent_id:
            self.store.get(request.parent_id)
        preview_date, preview_scope = None, None
        if kind == "cohort_preview":
            scope_config = (
                config.effective(
                    self.catalog.candidate_ids(request.dataset_id, config), {}
                )
                if isinstance(config, LabConfigV2)
                else config
            )
            preview_date, preview_scope = request.decision_date, request.scope
            if not day(config.start) <= day(preview_date) < day(config.end):
                raise ValueError(
                    "Preview date must lie inside configured backtest dates"
                )
            if (
                scope_config.scope == "per_asset"
                and preview_scope not in scope_config.coins
                or scope_config.scope == "pooled"
                and preview_scope is not None
            ):
                raise ValueError("Preview scope must match strategy scope")
        provenance.update(
            engine="copy_lab_v2" if isinstance(config, LabConfigV2) else "copy_lab_v1",
            source_hashes=source_fingerprint(),
            dependencies={
                name: version(name) for name in ("arblab", "duckdb", "pyarrow")
            },
        )
        return self.store.create(
            request.name,
            request.dataset_id,
            config.to_dict(),
            provenance,
            kind=kind,
            preview_date=preview_date,
            preview_scope=preview_scope,
            parent_id=request.parent_id,
        )

    def cancel(self, identifier):
        with self.lock:
            item = self.store.get(identifier)
            if item["status"] not in {"queued", "running"}:
                raise ValueError("Only queued or running jobs can be cancelled")
            if self.active == identifier and self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
                self.process = None
                self.active = None
            return self.store.transition(identifier, item["status"], "cancelled")

    def publish(self, item):
        staging = self.root / "staging" / item["id"]
        data = json.loads((staging / "completion.json").read_text())
        if not data["hashes"]:
            raise ValueError("Empty worker output")
        for name, expected in data["hashes"].items():
            path = (staging / name).resolve()
            if not path.is_relative_to(staging) or file_hash(path) != expected:
                raise ValueError("Worker artifact verification failed")
        output = self.root / "results" / item["id"]
        output.parent.mkdir(exist_ok=True)
        # Renames and final status are under the same coordinator lock as cancel.
        os.rename(staging, output)
        run_id = None
        if item["kind"] == "backtest":
            run_id = "hyperliquid_trader_ensemble_" + item["id"]
            (self.root / "reports").mkdir(exist_ok=True)
            os.rename(output / "report", self.root / "reports" / run_id)
        return run_id, data["hashes"]

    def tick(self):
        with self.lock:
            if self.process and self.process.poll() is not None:
                item = self.store.get(self.active)
                try:
                    if self.process.returncode:
                        raise ValueError("Worker rejected input or failed")
                    run_id, hashes = self.publish(item)
                    self.store.transition(
                        item["id"],
                        "running",
                        "completed",
                        run_id=run_id,
                        artifact_hashes=hashes,
                    )
                except Exception:
                    self.store.transition(
                        item["id"],
                        "running",
                        "failed",
                        error="Worker failed or rejected data; no validated result published",
                    )
                self.process, self.active = None, None
            if not self.process and not self.stop.is_set():
                item = self.store.next_queued()
                if item:
                    self.store.transition(item["id"], "queued", "running")
                    try:
                        if item["provenance"]["source_hashes"] != source_fingerprint():
                            raise ValueError("Engine changed since submission")
                        self.process = subprocess.Popen(
                            [
                                sys.executable,
                                "-m",
                                "hyperliquid_explorer_api.lab_worker",
                                "--root",
                                str(self.root),
                                "--id",
                                item["id"],
                            ],
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                        )
                        self.active = item["id"]
                    except Exception:
                        self.store.transition(
                            item["id"],
                            "running",
                            "failed",
                            error="Worker could not start or engine changed since submission",
                        )

    def loop(self):
        while not self.stop.wait(0.1):
            self.tick()
