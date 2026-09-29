from dataclasses import replace
from datetime import timedelta
import json
from pathlib import Path

import lz4.frame
import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.qualified_scheduled_activity import (
    QualifiedScheduledActivity,
)
from arblab.hyperliquid_copy.proxy_activity import ProxyActivity
from .test_candidate_day import resources
from .test_archive_job import inputs
from .test_prefix_qualification import qualify
from .test_qualified_scheduled_activity import scheduled_config

SEED_USER = "0x" + "ef" * 20
POLICY = "qualified_source_seed_v1"


@pytest.fixture
def seeded_source(tmp_path):
    from arblab.hyperliquid_copy.proxy_archive_download import download_archive
    from arblab.hyperliquid_copy.proxy_archive_import import import_archive
    from arblab.hyperliquid_copy.proxy_compact import compact_history

    _, source, _, _ = inputs(tmp_path, days=4)
    for key, body in source.bodies.items():
        if "20260801" in key:
            data = json.loads(lz4.frame.decompress(body))
            for pair in data["events"]:
                if pair[0] == "0x" + "ab" * 20:
                    pair[0] = SEED_USER
            source.bodies[key] = lz4.frame.compress(json.dumps(data).encode() + b"\n")
    raw = download_archive(source, "2026-08-01", "2026-08-05", tmp_path / "download")
    normalized = import_archive(
        raw, ["BTC"], tmp_path / "normalized", retain_boundary_spill=True
    )
    compact = compact_history(
        normalized, tmp_path / "compact", partitioning="source_day"
    )
    return qualify([compact], tmp_path)


def reader(resources, pin, **changes):
    kwargs = dict(
        coverage_start="2026-08-02",
        coverage_end="2026-08-04",
        semantics="gross_excludes_fee",
        history_policy=POLICY,
    )
    kwargs.update(changes)
    return QualifiedScheduledActivity(
        resources, pin, kwargs.pop("config", scheduled_config()), **kwargs
    )


def test_explicit_seed_policy_preserves_dormant_universe_and_positions(
    resources, seeded_source, tmp_path
):
    at = day("2026-08-03")
    report = json.loads(Path(seeded_source["path"]).read_text())
    config = scheduled_config().effective(["BTC"], {"BTC": 1})
    with ProxyActivity(
        [Path(row["path"]) for row in report["files"]], temp_root=tmp_path
    ) as ref:
        expected = ref.rank(at, config, "BTC", "gross_excludes_fee", smoke=True)
        position = ref.position(SEED_USER, "BTC", at)
    assert position is not None
    assert SEED_USER in {row["user"] for row in expected}
    with reader(resources, seeded_source) as actual:
        assert actual.origin == day("2026-08-01")
        assert actual.finish == day("2026-08-04")
        actual.prepare(at)
        assert actual.position(SEED_USER, "BTC", at) == position
        result = actual.rank(at, config, "BTC", "gross_excludes_fee", smoke=True)
        rows = [row for batch in result.iter_batches() for row in batch]
        assert {row["user"] for row in rows} == {row["user"] for row in expected}
        dormant = next(row for row in rows if row["user"] == SEED_USER)
        assert (
            not dormant["eligible"]
            and "no_activity_in_lookback" in dormant["exclusions"]
        )
    resources.lease.check()


@pytest.mark.parametrize(
    "change", ["policy", "left_padding", "right_padding", "warmup", "end"]
)
def test_invalid_seed_policy_or_bounds_reject(resources, seeded_source, change):
    kwargs = {
        "policy": dict(history_policy="implicit"),
        "left_padding": dict(coverage_start="2026-08-01"),
        "right_padding": dict(coverage_end="2026-08-05"),
        "warmup": dict(coverage_start="2026-08-03"),
        "end": dict(config=replace(scheduled_config(), end="2026-08-05")),
    }[change]
    before = resources.audit()
    with pytest.raises(ValueError):
        reader(resources, seeded_source, **kwargs)
    assert resources.audit() == before


def test_default_policy_still_rejects_interior_coverage(resources, seeded_source):
    with pytest.raises(ValueError, match="origin"):
        QualifiedScheduledActivity(
            resources,
            seeded_source,
            scheduled_config(),
            coverage_start="2026-08-02",
            coverage_end="2026-08-04",
            semantics="gross_excludes_fee",
        )


@pytest.mark.parametrize("field", ["history_policy", "coverage_start", "coverage_end"])
def test_seed_context_mutation_rejected(resources, seeded_source, field):
    with reader(resources, seeded_source) as actual:
        setattr(actual, field, "changed")
        with pytest.raises(ValueError):
            actual.prepare(day("2026-08-03"))


def test_seed_reader_never_uses_right_padding_for_queries(resources, seeded_source):
    with reader(resources, seeded_source) as actual:
        at = day("2026-08-04") - timedelta(hours=1)
        actual.prepare(at)
        with pytest.raises(ValueError):
            actual.hourly_exposure(
                SEED_USER,
                "BTC",
                at,
                day("2026-08-04") + timedelta(hours=1),
                max_price_age_seconds=86400,
            )
        with pytest.raises(ValueError):
            actual.prepare(day("2026-08-04"))
