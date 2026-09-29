import json

import pytest

from arblab.hyperliquid_copy.download import file_hash
from .test_proxy_market_bundle import bundle, segment


def verify(path, kind, tmp_path):
    from arblab.hyperliquid_copy.registration_market_inputs import (
        verified_market_bundle,
    )

    return verified_market_bundle(
        dict(path=str(path), sha256=file_hash(path)), kind=kind, temp_root=tmp_path
    )


@pytest.mark.parametrize("kind", ["prices", "funding"])
def test_bundle_rebuild_matches_pinned_source_evidence(tmp_path, kind):
    source = segment(tmp_path, "source", kind, range(4))
    path = bundle(tmp_path, kind, [source])
    result = verify(path, kind, tmp_path)
    assert result["manifest"] == json.loads(path.read_text())
    assert file_hash(result["file"]) == result["manifest"]["file"]["sha256"]
    assert not list(tmp_path.glob("registration_bundle_*"))


def test_changed_raw_evidence_rejected(tmp_path):
    source = segment(tmp_path, "source", "prices", range(4))
    path = bundle(tmp_path, "prices", [source])
    (source.parent / "source_0000.raw").write_bytes(b"changed")
    with pytest.raises(ValueError, match="identity|hash"):
        verify(path, "prices", tmp_path)


def test_rehashed_manifest_cannot_launder_false_source_completeness(tmp_path):
    source = segment(
        tmp_path, "source", "prices", range(4), missing=[0], complete=False
    )
    path = bundle(tmp_path, "prices", [source], begin=1)
    data = json.loads(path.read_text())
    data["source_manifests"][0]["complete"] = True
    path.write_text(json.dumps(data, sort_keys=True, indent=2))
    with pytest.raises(ValueError, match="reconstruction"):
        verify(path, "prices", tmp_path)


def test_wrong_input_kind_rejected(tmp_path):
    source = segment(tmp_path, "source", "prices", range(4))
    path = bundle(tmp_path, "prices", [source])
    with pytest.raises(ValueError, match="kind"):
        verify(path, "funding", tmp_path)
