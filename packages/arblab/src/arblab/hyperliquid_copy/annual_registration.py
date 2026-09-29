"""Offline annual publication gates; a qualified prefix is not a finished job."""

from dataclasses import replace
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import pyarrow.parquet as pq

from .archive import archive_keys
from .archive_job import ArchiveJob
from .archive_job_qualification import pin, verify
from .archive_job_store import STAGES
from .lab_config import day
from .feature_history_policy import checked_feature_policy
from .annual_execution_policy import checked_execution_policy
from .ranking_staging_policy import validate_policy as checked_staging_policy
from .prefix_qualification import _engine, _previous
from .archive_cache import _safe
from .download import file_hash
from .lab_config_proxy import LabConfigProxyScheduled
from .lab_pipeline_proxy import validate_proxy_run
from .proxy_dataset import ProxyDatasetManifest, SCHEMA
from .proxy_compact import _sync
from .registration_market_inputs import verified_market_bundle
from .registration_partitions import publish_interior, retain_full_source, free_space
from .qualified_registration import cache_reference as checked_cache_reference, POLICY
from .registration_provenance import NAMES
from .scheduled_activity import history_days


def completed_job_inputs(root, *, start, end):
    """Return verified padded source inputs, without asserting native availability.

    This does not download, retry, clean up, publish a dataset or reset spending.
    Raw payloads may already be disposed of; canonical content and retained raw
    provenance manifests must still verify against the frozen job's final report.
    """
    begin, finish = day(start), day(end)
    if not 0 < (finish - begin).days <= 732:
        raise ValueError("Invalid annual registration coverage interval")
    job = ArchiveJob(root)
    records = job.store.records()
    required = {(n, stage) for n in range(len(job.batches)) for stage in STAGES}
    if not required or set(records) != required:
        raise ValueError(
            "Archive job unfinished: every planned batch must be qualified"
        )
    index = len(job.batches) - 1
    report_pin = pin(records[index, "qualified"])
    verify(job.store, index, report_pin, recheck_content=True)
    report = _previous(report_pin, _engine())
    source_begin, source_end = day(report["source_start"]), day(report["source_end"])
    if (
        not source_begin + timedelta(days=1)
        <= begin
        < finish
        <= source_end - timedelta(days=1)
    ):
        raise ValueError(
            "Annual registration requires full source-day padding on both boundaries"
        )
    all_keys = [o["key"] for source in report["raw_sources"] for o in source["objects"]]
    expected = {
        key
        for n in range((source_end - source_begin).days)
        for key in archive_keys((source_begin + timedelta(days=n)).date().isoformat())
    }
    if len(all_keys) != len(expected) or set(all_keys) != expected:
        raise ValueError("Incomplete final archive source-key provenance")
    retained = {
        key
        for n in range((finish - begin).days)
        for key in archive_keys((begin + timedelta(days=n)).date().isoformat())
    }
    current = ArchiveJob(root)
    if current.store.frozen != job.store.frozen or current.store.records() != records:
        raise ValueError("Archive job identity changed during registration check")
    _previous(report_pin, _engine())
    return dict(
        qualification=report_pin,
        archive_engine=current.engine_evidence(),
        job_metadata_sha256=hashlib.sha256(job.store.frozen.encode()).hexdigest(),
        source_start=report["source_start"],
        source_end=report["source_end"],
        coverage_start=start,
        coverage_end=end,
        coins=list(report["coins"]),
        files=[
            {k: f[k] for k in ("path", "sha256", "bytes", "rows")}
            for f in report["files"]
        ],
        source_keys=sorted(retained),
        padding_source_keys=sorted(expected - retained),
        native_availability_qualified=False,
    )


def _publication_engine():
    return {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))}


def _file_entry(path):
    return dict(
        name=path.name,
        bytes=path.stat().st_size,
        rows=pq.ParquetFile(path).metadata.num_rows,
        sha256=file_hash(path),
    )


def _json_file(path, data, *, maximum=16 * 1024**2):
    # The semantic-hash serializer intentionally stringifies decimals; persisted
    # UI configuration must preserve JSON numeric types instead.
    content = json.dumps(
        data, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    if len(content) > maximum:
        raise ValueError("Registration metadata byte bound exceeded")
    with path.open("xb") as stream:
        stream.write(content)
    _sync(path)


def _copy_evidence(pin, destination):
    source = Path(pin["path"])
    _safe(source)
    if not source.is_file() or source.stat().st_size > 16 * 1024**2:
        raise ValueError("Registration source evidence byte bound exceeded")
    if file_hash(source) != pin["sha256"]:
        raise ValueError("Registration source evidence changed")
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=1024 * 1024)
    if file_hash(destination) != pin["sha256"] or file_hash(source) != pin["sha256"]:
        raise ValueError("Registration source evidence changed while copying")
    _sync(destination)


def register_annual_dataset(
    job_root,
    price_pin,
    funding_pin,
    target,
    *,
    config,
    name,
    cache_reference=None,
    feature_history_policy=None,
    ranking_staging_policy=None,
    execution_policy_name=None,
):
    """Publish a completed, evidence-qualified corpus; no remote IO or orders.

    A short fixture may exercise this path, but actual annual acceptance requires
    the caller to execute and save the full requested weekly/daily comparison.
    """
    # Local import keeps the evidence factory's use of completed_job_inputs acyclic
    # during module initialization. Both functions operate on frozen source pins.
    from .registration_native_history import qualify_observed_history

    checked_feature_policy(feature_history_policy)
    checked_staging_policy(ranking_staging_policy)
    if ranking_staging_policy is not None and cache_reference is None:
        raise ValueError("Ranking staging policy requires an explicit shared cache")
    if feature_history_policy is not None and cache_reference is None:
        raise ValueError("Feature history policy requires an explicit shared cache")
    if not isinstance(config, LabConfigProxyScheduled):
        raise ValueError("Annual registration requires scheduled configuration")
    if not isinstance(name, str) or not name.strip() or len(name) > 1000:
        raise ValueError("Expected bounded dataset name")
    reference = (
        None if cache_reference is None else checked_cache_reference(cache_reference)
    )
    annual_policy = checked_execution_policy(
        execution_policy_name,
        feature_history_policy=feature_history_policy,
        ranking_staging_policy=ranking_staging_policy,
        registration_policy=POLICY if reference is not None else None,
        cache_reference=reference,
    )
    target = Path(target).absolute()
    _safe(target)
    if ".." in target.parts or target.exists():
        raise ValueError("Dataset target already exists or is unsafe")
    engine = _publication_engine()
    start = (
        (day(config.start) - timedelta(days=history_days(config))).date().isoformat()
    )
    end = config.end
    inputs = completed_job_inputs(job_root, start=start, end=end)
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix=".annual_registration_", dir=target.parent
    ) as scratch:
        scratch = Path(scratch)
        stage = scratch / "dataset"
        stage.mkdir()
        prices = verified_market_bundle(price_pin, kind="prices", temp_root=scratch)
        funding = verified_market_bundle(funding_pin, kind="funding", temp_root=scratch)
        if set(prices["manifest"]["intervals"]) != set(inputs["coins"]):
            raise ValueError("Price/native market scope mismatch")
        evidence = qualify_observed_history(
            job_root, funding_pin, start=start, end=end, temp_root=scratch
        )
        if evidence["inputs"] != inputs:
            raise ValueError("Native evidence source changed")
        free_space(
            stage, prices["file"].stat().st_size + funding["file"].stat().st_size
        )
        files = (
            publish_interior(inputs["files"], day(start), day(end), stage)
            if reference is None
            else retain_full_source(inputs["files"], stage)
        )
        for source, filename in (
            (prices["file"], "bars.parquet"),
            (funding["file"], "funding.parquet"),
        ):
            destination = stage / filename
            shutil.copyfile(source, destination)
            if file_hash(destination) != file_hash(source):
                raise ValueError("Market input changed while copying")
            _sync(destination)
            files.append(_file_entry(destination))
        evidence_path = stage / "native_history_evidence.json"
        _json_file(evidence_path, evidence)
        evidence_hash = file_hash(evidence_path)
        metadata = dict(
            schema=SCHEMA,
            validation_mode="sharded_v1",
            synthetic=False,
            name=name,
            coverage_start=start,
            coverage_end=end,
            coins=inputs["coins"],
            fee_semantics="gross_excludes_fee",
            mappings=prices["manifest"]["mappings"],
            coverage_note="All-wallet scoped native activity; padded interior event-time history with conservative observed-history starts. Proxy-priced follower, not exact execution or complete wallet equity.",
            default_config=config.to_dict(),
            activity_provenance=dict(
                scope="all_wallets_for_declared_markets",
                complete=True,
                source_keys=inputs["source_keys"],
                padding_source_keys=inputs["padding_source_keys"],
                source_manifest_hash=inputs["qualification"]["sha256"],
                boundary_policy="Complete source-day padding; exact interior projection preserves all economic rows and duplicates",
            ),
            price_source_manifest_hash=price_pin["sha256"],
            funding_source_manifest_hash=funding_pin["sha256"],
            price_policy=prices["manifest"]["price_policy"],
            files=files,
            native_history_policy=evidence["policy"],
            native_history_evidence=dict(name=evidence_path.name, sha256=evidence_hash),
            native_history=[
                dict(
                    instrument_id=coin,
                    available_from=at,
                    evidence_sha256=evidence_hash,
                    description="Conservative observed history under complete scoped source coverage; not a listing date",
                )
                for coin, at in sorted(evidence["starts"].items())
            ],
            funding_history=[
                dict(
                    instrument_id=coin,
                    available_from=evidence["funding_starts"][coin],
                    available_until=evidence["funding_ends"][coin],
                    evidence_sha256=evidence_hash,
                    description="Verified retained funding coverage; distinct from observed activity history",
                )
                for coin in sorted(evidence["starts"])
            ],
            registration_engine=engine,
        )
        if reference is not None:
            metadata.update(
                validation_mode="qualified_v1",
                history_policy=POLICY,
                derived_cache=reference,
                coverage_note="All-wallet scoped native activity; full canonical source history retained for candidate and position seeds. Original qualified archive paths and the explicitly shared derived cache are durable dependencies. Proxy-priced follower, not exact execution or complete wallet equity.",
            )
            metadata["activity_provenance"]["boundary_policy"] = (
                "Complete source-day padding; byte-identical full canonical retention including boundary spill; causal queries use registered coverage and exact metric lookback"
            )
            if feature_history_policy is not None:
                metadata["feature_history_policy"] = feature_history_policy
            if ranking_staging_policy is not None:
                metadata["ranking_staging_policy"] = ranking_staging_policy
            if annual_policy is not None:
                metadata["execution_policy"] = annual_policy.name
        for source_pin, filename in (
            (price_pin, "price_source.json"),
            (funding_pin, "funding_source.json"),
            (inputs["qualification"], "activity_qualification.json"),
        ):
            _copy_evidence(source_pin, stage / filename)
        _json_file(stage / "activity_source.json", inputs)
        metadata["registration_provenance"] = {
            n: file_hash(stage / n) for n in sorted(NAMES)
        }
        if reference is not None:
            from .derived_cache_lease import CacheLease
            from .qualified_registered_activity import _open_resources
            from .qualified_source_session import QualifiedSourceSession
            from .registration_capacity import export_registration_capacity

            with CacheLease(reference["path"]) as lease:
                resources = _open_resources(lease, reference)
                metadata["candidate_capacity"] = export_registration_capacity(
                    resources, QualifiedSourceSession(inputs["qualification"]), stage
                )
        _json_file(stage / "manifest.json", metadata, maximum=1_000_000)
        manifest = ProxyDatasetManifest(stage)
        with manifest.load(
            temp_root=scratch, expected_hash=manifest.identity, config=config
        ) as loaded:
            for cadence in ("weekly", "daily"):
                matched = replace(
                    config,
                    rebalance=cadence,
                    trader=replace(config.trader, reselection=cadence),
                    market_universe=replace(
                        config.market_universe, reselection=cadence
                    ),
                )
                validate_proxy_run(loaded, matched, annual=True)
        cache = stage / ".activity_checkpoints"
        if cache.exists():
            cache.rename(scratch / "validation_checkpoints")
        # Cached absolute staging paths must never survive dataset publication.
        if completed_job_inputs(job_root, start=start, end=end) != inputs:
            raise ValueError("Archive inputs changed before publication")
        for pin, kind in ((price_pin, "prices"), (funding_pin, "funding")):
            verified_market_bundle(pin, kind=kind, temp_root=scratch)
        manifest.verify()
        if _publication_engine() != engine:
            raise ValueError("Registration engine changed before publication")
        _sync(stage)
        if target.exists() or target.is_symlink():
            raise ValueError("Dataset target already exists")
        stage.rename(target)
        _sync(target.parent)
    return target / "manifest.json"
