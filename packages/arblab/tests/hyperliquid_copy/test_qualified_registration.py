from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from arblab.hyperliquid_copy.download import file_hash
from arblab.hyperliquid_copy.lab_config import day
from .test_candidate_day import resources
from .test_annual_registration import publication_inputs


@pytest.fixture
def registered_source(tmp_path, resources):
    from arblab.hyperliquid_copy.annual_registration import register_annual_dataset

    job, _, config, target, args = publication_inputs(tmp_path)
    register_annual_dataset(*args, config=config, name="Qualified loader fixture")
    metadata = json.loads((target / "manifest.json").read_text())
    report = json.loads((target / "activity_qualification.json").read_text())
    retained = []
    for i, source in enumerate(report["files"]):
        name = f"fills-{i:04d}.parquet"
        destination = target / name
        # Legacy registration may hardlink wholly interior fixture files.
        # Unlink this owned test destination before writing, preserving source.
        destination.unlink()
        shutil.copyfile(source["path"], destination)
        retained.append(
            dict(name=name, **{k: source[k] for k in ("bytes", "rows", "sha256")})
        )
    metadata["files"] = retained + [
        r for r in metadata["files"] if not r["name"].startswith("fills-")
    ]
    metadata.update(
        validation_mode="qualified_v1",
        history_policy="qualified_source_seed_v1",
        derived_cache=dict(path=str(resources.root), identity=resources.identity),
    )
    (target / "manifest.json").write_text(json.dumps(metadata))
    return SimpleNamespace(
        directory=target,
        metadata=metadata,
        paths={r["name"]: target / r["name"] for r in metadata["files"]},
        coverage_start=day(metadata["coverage_start"]),
        coverage_end=day(metadata["coverage_end"]),
    ), config


def verify(manifest):
    from arblab.hyperliquid_copy.qualified_registration import (
        verify_qualified_registration,
    )

    return verify_qualified_registration(manifest)


def test_unknown_feature_history_policy_rejects_registration_and_manifest(
    registered_source, resources
):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest

    manifest, _ = registered_source
    manifest.metadata["feature_history_policy"] = "unknown"
    before = resources.audit()
    with pytest.raises(ValueError, match="feature.*policy"):
        verify(manifest)
    (manifest.directory / "manifest.json").write_text(json.dumps(manifest.metadata))
    with pytest.raises(ValueError, match="feature.*policy"):
        ProxyDatasetManifest(manifest.directory)
    assert resources.audit() == before


def test_registered_source_binding_returns_original_pin_and_existing_cache(
    registered_source, resources
):
    manifest, _ = registered_source
    before = resources.audit()
    pin, cache = verify(manifest)
    inputs = json.loads((manifest.directory / "activity_source.json").read_text())
    assert pin == inputs["qualification"]
    assert cache == dict(path=str(resources.root), identity=resources.identity)
    assert resources.audit() == before
    # Results must not alias operator metadata.
    cache["identity"] = "changed"
    assert verify(manifest)[1]["identity"] == resources.identity


@pytest.mark.parametrize(
    "fault",
    [
        "policy",
        "mode",
        "coverage",
        "source_pin",
        "files",
        "order",
        "copy",
        "source",
        "cache",
        "cache_relative",
        "cache_missing",
    ],
)
def test_invalid_registered_bindings_reject_without_cache_writes(
    registered_source, resources, fault
):
    manifest, _ = registered_source
    before = resources.audit()
    data = manifest.metadata
    if fault == "policy":
        data["history_policy"] = "implicit"
    elif fault == "mode":
        data["validation_mode"] = "sharded_v1"
    elif fault == "coverage":
        data["coverage_start"] = "2026-08-01"
    elif fault == "source_pin":
        data["activity_provenance"]["source_manifest_hash"] = "0" * 64
    elif fault == "files":
        data["files"] = data["files"][1:]
    elif fault == "order":
        data["files"][0], data["files"][1] = data["files"][1], data["files"][0]
    elif fault in ("copy", "source"):
        path = manifest.paths[data["files"][0]["name"]]
        if fault == "source":
            report = json.loads(
                (manifest.directory / "activity_qualification.json").read_text()
            )
            path = Path(report["files"][0]["path"])
        with path.open("ab") as stream:
            stream.write(b"changed")
    elif fault == "cache":
        data["derived_cache"]["limit_bytes"] = 16 * 1024**3
    elif fault == "cache_relative":
        data["derived_cache"]["path"] = "relative"
    else:
        data["derived_cache"]["path"] = str(manifest.directory / "missing-cache")
    with pytest.raises(ValueError):
        verify(manifest)
    assert resources.audit() == before


@pytest.mark.parametrize(
    "sidecar", ["activity_source.json", "activity_qualification.json"]
)
def test_sidecar_rebinding_cannot_hide_wrong_source(registered_source, sidecar):
    manifest, _ = registered_source
    path = manifest.directory / sidecar
    data = json.loads(path.read_text())
    data["source_start"] = "2026-07-31"
    path.write_text(json.dumps(data))
    manifest.metadata["registration_provenance"][sidecar] = file_hash(path)
    with pytest.raises(ValueError):
        verify(manifest)


def test_final_source_recheck_detects_mutation_during_destination_verification(
    registered_source, monkeypatch
):
    from arblab.hyperliquid_copy.qualified_day import QualifiedFile

    manifest, _ = registered_source
    report = json.loads(
        (manifest.directory / "activity_qualification.json").read_text()
    )
    original_path = Path(report["files"][0]["path"])
    original = QualifiedFile.verify

    def changed(entry):
        result = original(entry)
        if entry.path.parent == manifest.directory and entry.path.name.startswith(
            "fills-"
        ):
            with original_path.open("ab") as stream:
                stream.write(b"changed")
        return result

    monkeypatch.setattr(QualifiedFile, "verify", changed)
    with pytest.raises(ValueError):
        verify(manifest)
