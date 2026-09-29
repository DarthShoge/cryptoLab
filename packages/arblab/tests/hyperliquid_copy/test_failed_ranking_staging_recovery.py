"""Failed staging recovery tests."""  # ruff: noqa: F811

import pytest

from arblab.hyperliquid_copy.ranking_staging_owner import RankingStagingOwner
from .test_candidate_day import resources  # noqa: F401


def failed_staging(resources):
    context = {"query": {"decision": "2026-10-06", "scope": "BTC"}}
    owner = RankingStagingOwner.create(resources, context)
    manifest = owner._allocations["manifest"]["path"]
    with owner.path("metrics").open("xb") as stream:
        stream.write(b"partial metrics")
    nested = owner.path("scratch") / "partition"
    nested.mkdir()
    (nested / "rows.parquet").write_bytes(b"partial scratch")
    owner.close()
    return context, manifest


def test_recovery_releases_failed_outputs_and_retains_manifest_receipt(resources):
    from arblab.hyperliquid_copy.failed_ranking_staging_recovery import recover

    context, manifest = failed_staging(resources)
    before = resources.audit()
    result = recover(resources, manifest, context)

    assert result["released_roles"] == ["scratch", "metrics", "scores", "ranking"]
    assert result["receipt"].path == manifest
    assert (resources.root / manifest).exists()
    assert resources.audit()["reserved_bytes"] == 0
    assert resources.audit()["retained_bytes"] < before["reserved_bytes"]
    assert recover(resources, manifest, context) == result


@pytest.mark.parametrize("change", ["context", "manifest", "ledger", "symlink"])
def test_recovery_rejects_changed_authority_before_mutation(
    resources, change, tmp_path
):
    from arblab.hyperliquid_copy.failed_ranking_staging_recovery import recover

    context, manifest = failed_staging(resources)
    manifest_path = resources.root / manifest
    expected = context
    if change == "context":
        expected = {"query": "different"}
    elif change == "manifest":
        manifest_path.write_bytes(manifest_path.read_bytes() + b"changed")
    elif change == "ledger":
        with resources._connect() as db, db:
            db.execute(
                "UPDATE allocations SET maximum=maximum-1 WHERE path LIKE 'staging/%.parquet'"
            )
    else:
        scratch = next((resources.root / "scratch").iterdir())
        target = tmp_path / "outside"
        target.write_bytes(b"outside")
        (scratch / "unsafe").symlink_to(target)
    before = resources.audit if change != "symlink" else None
    with pytest.raises(ValueError):
        recover(resources, manifest, expected)
    assert manifest_path.exists()
    if before is not None:
        # The recovery itself did not remove any manifest-owned path.
        assert any((resources.root / name).iterdir() for name in ("scratch", "staging"))
