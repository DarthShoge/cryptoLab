import json

import pytest

from arblab.hyperliquid_copy.qualified_registration import cache_reference
from .test_candidate_day import resources


def test_reference_validates_existing_identity_without_taking_lease(resources):
    reference = dict(path=str(resources.root), identity=resources.identity)
    before = resources.audit()
    assert cache_reference(reference) == reference
    assert resources.audit() == before


@pytest.mark.parametrize(
    "fault",
    [
        "identity",
        "uninitialized",
        "missing_catalog",
        "marker",
        "aliased_marker",
        "symlink",
    ],
)
def test_reference_rejects_uninitialized_or_changed_cache(resources, tmp_path, fault):
    reference = dict(path=str(resources.root), identity=resources.identity)
    if fault == "identity":
        reference["identity"] = "another-cache"
    elif fault == "uninitialized":
        root = tmp_path / "empty"
        root.mkdir()
        reference["path"] = str(root)
    elif fault == "missing_catalog":
        (resources.root / "resources.sqlite3").rename(resources.root / "moved.sqlite3")
    elif fault == "marker":
        marker = resources.root / ".initialized"
        data = json.loads(marker.read_text())
        data["limit_bytes"] *= 2
        marker.write_text(json.dumps(data))
    elif fault == "aliased_marker":
        import os

        os.link(resources.root / ".initialized", tmp_path / "aliased-marker")
    else:
        root = tmp_path / "alias"
        root.symlink_to(resources.root, target_is_directory=True)
        reference["path"] = str(root)
    with pytest.raises(ValueError):
        cache_reference(reference)
    if fault == "uninitialized":
        assert list(root.iterdir()) == []
