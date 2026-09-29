import json

import pytest

from arblab.hyperliquid_copy.lab_config import day
from .test_candidate_day import resources
from .test_qualified_day import qualified
from .test_annual_registration import publication_inputs


@pytest.fixture
def registered(resources, tmp_path):
    from arblab.hyperliquid_copy.annual_registration import register_annual_dataset

    _, _, config, target, args = publication_inputs(tmp_path)
    reference = dict(path=str(resources.root), identity=resources.identity)
    resources.lease.__exit__(None, None, None)
    register_annual_dataset(
        *args, config=config, name="Capacity integration", cache_reference=reference
    )
    return target, config


def test_qualified_registration_retains_capacity_without_query_on_reopen(
    registered, monkeypatch
):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest
    from arblab.hyperliquid_copy import candidate_capacity as producer

    target, config = registered
    metadata = json.loads((target / "manifest.json").read_text())
    assert "candidate_capacity" in metadata

    def forbidden(*args, **kwargs):
        pytest.fail("registered capacity reopened a raw query")

    monkeypatch.setattr(producer, "_write", forbidden)
    from arblab.hyperliquid_copy.registration_capacity import load_registration_capacity

    for _ in range(2):
        manifest = ProxyDatasetManifest(target)
        evidence = load_registration_capacity(target, manifest.metadata)
        assert evidence.upper_bound(day(config.start), ["BTC"], "BTC") == 2
        assert evidence.upper_bound(day(config.start), ["BTC"], None) == 2
    path = target / "candidate_capacity.parquet"
    with path.open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError):
        ProxyDatasetManifest(target)


@pytest.mark.parametrize("fault", ["scope", "source", "publication", "path", "missing"])
def test_registered_capacity_rejects_substituted_provenance(registered, fault):
    from arblab.hyperliquid_copy.registration_capacity import load_registration_capacity
    from arblab.hyperliquid_copy.download import file_hash

    target, _ = registered
    metadata = json.loads((target / "manifest.json").read_text())
    path = target / "candidate_capacity.json"
    value = json.loads(path.read_text())
    if fault == "scope":
        value["inputs"]["source"]["coins"] = []
    elif fault == "source":
        value["inputs"]["source"]["membership_sha256"] = "0" * 64
    elif fault == "publication":
        value["publication"]["key"] = "0" * 64
    elif fault == "path":
        value["payload"]["name"] = "../candidate_capacity.parquet"
    else:
        (target / "candidate_capacity.parquet").rename(target / "missing.parquet")
    path.write_text(json.dumps(value))
    metadata["candidate_capacity"].update(
        bytes=path.stat().st_size, sha256=file_hash(path)
    )
    with pytest.raises(ValueError):
        load_registration_capacity(target, metadata)


def test_legacy_absence_remains_explicit(registered):
    from arblab.hyperliquid_copy.registration_capacity import load_registration_capacity

    target, _ = registered
    metadata = json.loads((target / "manifest.json").read_text())
    metadata.pop("candidate_capacity")
    assert load_registration_capacity(target, metadata) is None


def test_batch_capacity_verifies_once_at_each_boundary(registered, monkeypatch):
    from arblab.hyperliquid_copy.registration_capacity import load_registration_capacity
    from arblab.hyperliquid_copy.capacity_schedule import ranking_upper_bound

    target, config = registered
    metadata = json.loads((target / "manifest.json").read_text())
    evidence = load_registration_capacity(target, metadata)
    expected = ranking_upper_bound(config, ["BTC"], evidence.upper_bound)
    calls = []
    verify = evidence._verify

    def counted():
        calls.append(1)
        verify()

    monkeypatch.setattr(evidence, "_verify", counted)
    assert evidence.ranking_rows(config, ["BTC"]) == expected
    assert len(calls) == 2


@pytest.mark.parametrize("fault", ["schedule_engine", "initial_coins"])
def test_batch_capacity_preserves_initial_context(registered, monkeypatch, fault):
    from arblab.hyperliquid_copy import registration_capacity as module

    target, config = registered
    metadata = json.loads((target / "manifest.json").read_text())
    evidence = module.load_registration_capacity(target, metadata)
    coins = ["BTC"]
    if fault == "schedule_engine":
        original = module.file_hash
        monkeypatch.setattr(
            module,
            "file_hash",
            lambda path: "0" * 64
            if path.name == "capacity_schedule.py"
            else original(path),
        )
    else:
        original = evidence._verify

        def changed():
            coins.clear()
            original()

        monkeypatch.setattr(evidence, "_verify", changed)
    with pytest.raises(ValueError):
        evidence.ranking_rows(config, coins)


def test_batch_capacity_rejects_mutation_during_calculation(registered, monkeypatch):
    from arblab.hyperliquid_copy import registration_capacity as module
    from arblab.hyperliquid_copy import capacity_schedule

    target, config = registered
    metadata = json.loads((target / "manifest.json").read_text())
    evidence = module.load_registration_capacity(target, metadata)
    original = capacity_schedule.ranking_upper_bound

    def changed(*args):
        result = original(*args)
        with (target / "activity_source.json").open("ab") as handle:
            handle.write(b"late mutation")
        return result

    monkeypatch.setattr(capacity_schedule, "ranking_upper_bound", changed)
    with pytest.raises(ValueError):
        evidence.ranking_rows(config, ["BTC"])


@pytest.mark.parametrize("fault", ["provenance", "source_engine", "reader_engine"])
def test_retained_capacity_guards_final_context(registered, monkeypatch, fault):
    from arblab.hyperliquid_copy import registration_capacity as module

    target, config = registered
    metadata = json.loads((target / "manifest.json").read_text())
    evidence = module.load_registration_capacity(target, metadata)
    original = module.capacity_engine

    def changed():
        result = original()
        with (target / "activity_source.json").open("ab") as handle:
            handle.write(b"late change")
        return result

    if fault == "provenance":
        monkeypatch.setattr(module, "capacity_engine", changed)
    elif fault == "source_engine":
        monkeypatch.setattr(module, "source_engine", lambda: "changed")
    else:
        from pathlib import Path

        original_hash = module.file_hash
        monkeypatch.setattr(
            module,
            "file_hash",
            lambda path: "0" * 64
            if path == Path(module.__file__)
            else original_hash(path),
        )
    with pytest.raises(ValueError):
        evidence.upper_bound(day(config.start), ["BTC"], None)


def test_export_rejects_descriptor_change_during_final_check(
    resources, qualified, tmp_path, monkeypatch
):
    from arblab.hyperliquid_copy import registration_capacity as module
    from arblab.hyperliquid_copy.qualified_source_session import QualifiedSourceSession

    stage = tmp_path / "unpublished"
    stage.mkdir()
    original = module.CandidateCapacity._verify_artifact

    def changed(counts):
        original(counts)
        descriptor = stage / module.DESCRIPTOR
        if descriptor.exists():
            with descriptor.open("ab") as handle:
                handle.write(b" ")

    monkeypatch.setattr(module.CandidateCapacity, "_verify_artifact", changed)
    with pytest.raises(ValueError):
        module.export_registration_capacity(
            resources, QualifiedSourceSession(qualified), stage
        )
