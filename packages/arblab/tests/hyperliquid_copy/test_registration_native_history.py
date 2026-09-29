from pathlib import Path
import json

import pytest

from arblab.hyperliquid_copy.archive_job import ArchiveJob
from arblab.hyperliquid_copy.download import file_hash
from .test_archive_job import inputs
from .test_proxy_market_bundle import segment, bundle


def prepare(tmp_path, *, funding_begin=-24, coins=("BTC",)):
    inventory, source, total, _ = inputs(tmp_path, days=4)
    job = ArchiveJob.create(
        inventory, tmp_path / "job", list(coins), max_download_bytes=total
    )
    job.run_next(source=source)
    funding = segment(tmp_path, "funding_source", "funding", range(funding_begin, 24))
    funding_bundle = bundle(tmp_path, "funding", [funding], begin=funding_begin, end=24)
    return job, dict(path=str(funding_bundle), sha256=file_hash(funding_bundle))


def qualify(job, funding_pin, tmp_path):
    from arblab.hyperliquid_copy.registration_native_history import (
        qualify_observed_history,
    )

    return qualify_observed_history(
        job.store.root,
        funding_pin,
        start="2026-08-02",
        end="2026-08-04",
        temp_root=tmp_path,
    )


def test_completed_source_plus_funding_qualifies_observed_not_listing_history(tmp_path):
    job, funding = prepare(tmp_path)
    report = qualify(job, funding, tmp_path)
    assert report["policy"] == "complete_source_observed_history_v2"
    assert report["listing_dates_verified"] is False
    assert report["starts"] == {"BTC": "2026-08-02T00:00:00+00:00"}
    assert (
        report["observed"]["markets"]["BTC"]["first_event"]
        == "2026-08-01T00:00:00+00:00"
    )
    assert report["funding_bundle"] == funding
    assert report["funding_starts"] == {"BTC": "2026-08-02T00:00:00+00:00"}
    assert report["funding_ends"] == {"BTC": "2026-08-04T00:00:00+00:00"}


def test_missing_initial_funding_hour_is_not_filled_or_native_start_shifted(tmp_path):
    job, funding = prepare(tmp_path, funding_begin=-23)
    with pytest.raises(ValueError, match="funding.*native|native.*funding"):
        qualify(job, funding, tmp_path)


def test_partial_first_activity_hour_keeps_distinct_later_funding_start(
    tmp_path, monkeypatch
):
    import arblab.hyperliquid_copy.registration_native_history as module

    qualification = dict(path="qualified/manifest.json", sha256="a" * 64)
    inputs = dict(
        coins=["xyz:TSLA"],
        qualification=qualification,
        coverage_start="2025-09-01",
        coverage_end="2026-09-01",
    )
    observed = dict(
        qualification=qualification,
        markets={
            "xyz:TSLA": dict(
                rows=1,
                first_event="2025-11-13T14:31:56.848000+00:00",
                last_event="2026-08-31T23:59:00+00:00",
            )
        },
    )
    funding = dict(
        intervals={
            "xyz:TSLA": (
                "2025-11-13T15:00:00+00:00",
                "2026-09-01T00:00:00+00:00",
            )
        }
    )
    monkeypatch.setattr(module, "completed_job_inputs", lambda *a, **k: inputs)
    monkeypatch.setattr(module, "scan_qualified_history", lambda *a, **k: observed)
    monkeypatch.setattr(
        module,
        "verified_market_bundle",
        lambda *a, **k: dict(manifest=funding),
    )

    result = module.qualify_observed_history(
        tmp_path / "job",
        dict(path="funding/manifest.json", sha256="b" * 64),
        start="2025-09-01",
        end="2026-09-01",
        temp_root=tmp_path,
    )

    assert result["policy"] == "complete_source_observed_history_v2"
    assert result["schema"] == "hyperliquid_native_history_evidence_v2"
    assert result["starts"] == {"xyz:TSLA": "2025-11-13T14:00:00+00:00"}
    assert result["funding_starts"] == {
        "xyz:TSLA": "2025-11-13T15:00:00+00:00"
    }
    assert result["funding_ends"] == {
        "xyz:TSLA": "2026-09-01T00:00:00+00:00"
    }


def test_absent_declared_market_cannot_receive_an_availability_date(tmp_path):
    job, funding = prepare(tmp_path, coins=("BTC", "UNOBSERVED"))
    with pytest.raises(ValueError, match="absent|unobserved|scope"):
        qualify(job, funding, tmp_path)


@pytest.mark.parametrize("fault", [None, "file", "start", "scope", "missing"])
def test_loader_verifies_native_history_sidecar_and_bindings(tmp_path, fault):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from .test_proxy_dataset import registered

    metadata = registered(tmp_path)
    qualification = dict(path="synthetic/manifest.json", sha256="c" * 64)
    evidence = dict(
        schema="hyperliquid_native_history_evidence_v1",
        policy="complete_source_observed_history_v1",
        listing_dates_verified=False,
        native_availability_qualified=True,
        research_eligible=False,
        starts={"BTC": "2026-08-01T00:00:00+00:00"},
        coverage_start=metadata["coverage_start"],
        coverage_end=metadata["coverage_end"],
        inputs=dict(coins=["BTC"], qualification=qualification),
        observed=dict(qualification=qualification),
        funding_bundle=dict(path="funding/manifest.json", sha256="b" * 64),
    )
    if fault == "scope":
        evidence["starts"] = {"ETH": "2026-08-01T00:00:00+00:00"}
    sidecar = tmp_path / "native_history_evidence.json"
    sidecar.write_text(json.dumps(evidence))
    digest = file_hash(sidecar)
    metadata.update(
        native_history_policy=evidence["policy"],
        native_history_evidence=dict(name=sidecar.name, sha256=digest),
        native_history=[
            dict(
                instrument_id="BTC",
                available_from="2026-08-01T00:00:00+00:00",
                evidence_sha256=digest,
                description="Synthetic bound evidence fixture",
            )
        ],
    )
    metadata["activity_provenance"]["source_manifest_hash"] = "c" * 64
    if fault == "start":
        metadata["native_history"][0]["available_from"] = "2026-08-02T00:00:00+00:00"
    if fault == "missing":
        metadata.pop("native_history_evidence")
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    if fault == "file":
        sidecar.write_text(sidecar.read_text() + " ")
    if fault:
        with pytest.raises(ValueError, match="evidence|sidecar"):
            ProxyDatasetManifest(tmp_path)
    else:
        ProxyDatasetManifest(tmp_path).verify()


def test_loader_binds_distinct_funding_history_from_v2_evidence(tmp_path):
    from datetime import datetime

    from arblab.hyperliquid_copy.lab_config import day
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from .test_proxy_dataset import registered

    metadata = registered(tmp_path)
    qualification = dict(path="synthetic/manifest.json", sha256="c" * 64)
    evidence = dict(
        schema="hyperliquid_native_history_evidence_v2",
        policy="complete_source_observed_history_v2",
        listing_dates_verified=False,
        native_availability_qualified=True,
        research_eligible=False,
        starts={"BTC": "2026-08-01T00:00:00+00:00"},
        funding_starts={"BTC": "2026-08-01T01:00:00+00:00"},
        funding_ends={"BTC": "2026-08-04T00:00:00+00:00"},
        coverage_start=metadata["coverage_start"],
        coverage_end=metadata["coverage_end"],
        inputs=dict(coins=["BTC"], qualification=qualification),
        observed=dict(qualification=qualification),
        funding_bundle=dict(path="funding/manifest.json", sha256="b" * 64),
    )
    sidecar = tmp_path / "native_history_evidence.json"
    sidecar.write_text(json.dumps(evidence))
    digest = file_hash(sidecar)
    metadata.update(
        native_history_policy=evidence["policy"],
        native_history_evidence=dict(name=sidecar.name, sha256=digest),
        native_history=[
            dict(
                instrument_id="BTC",
                available_from=evidence["starts"]["BTC"],
                evidence_sha256=digest,
                description="Synthetic bound evidence fixture",
            )
        ],
        funding_history=[
            dict(
                instrument_id="BTC",
                available_from=evidence["funding_starts"]["BTC"],
                available_until=evidence["funding_ends"]["BTC"],
                evidence_sha256=digest,
                description="Synthetic funding coverage fixture",
            )
        ],
    )
    metadata["activity_provenance"]["source_manifest_hash"] = "c" * 64
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))

    manifest = ProxyDatasetManifest(tmp_path)

    assert manifest.native_starts == {"BTC": day("2026-08-01")}
    assert manifest.funding_starts == {
        "BTC": datetime.fromisoformat("2026-08-01T01:00:00+00:00")
    }
