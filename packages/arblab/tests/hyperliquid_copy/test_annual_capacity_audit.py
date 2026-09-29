from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


def module():
    path = (
        Path(__file__).resolve().parents[4]
        / "tools/audit_hyperliquid_annual_capacity.py"
    )
    spec = importlib.util.spec_from_file_location("annual_capacity_tool", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Source:
    coins = ("BTC",)

    def inputs(self):
        return {"pin": {"sha256": "a" * 64}}


class Resources:
    limit = 64 * 1024**3

    def __init__(self, root):
        self.root = root

    def audit(self):
        return {
            "retained_bytes": 5 * 1024**3,
            "reserved_bytes": 0,
            "metadata_bytes": 32 * 1024**2,
            "total_bytes": 5 * 1024**3 + 32 * 1024**2,
        }


class Counts:
    def __init__(self, value):
        self.value = value

    def upper_bound(self, *_):
        return self.value


def test_capacity_result_uses_bounded_policy_model_and_rejects_excess_rows(
    tmp_path, monkeypatch
):
    from .test_proxy_weekly import weekly_config

    audit_module = module()

    weekly = weekly_config(end="2026-08-08")
    daily = replace(
        weekly,
        rebalance="daily",
        trader=replace(weekly.trader, reselection="daily"),
        market_universe=replace(weekly.market_universe, reselection="daily"),
    )
    monkeypatch.setattr(
        audit_module.shutil,
        "disk_usage",
        lambda _: SimpleNamespace(free=1024**4),
    )
    result = audit_module._result(
        Source(),
        Resources(tmp_path),
        {"identity": "x"},
        weekly,
        daily,
        {},
        Counts(250_000_001),
    )
    assert not result["admitted"]
    assert (
        "annual_feature_retention_has_no_enforced_cumulative_byte_bound"
        not in result["blocking_bounds"]
    )
    assert any("ranking_rows_exceed" in item for item in result["blocking_bounds"])
    assert result["output_disk"]["ranking_copy_count"] == 5
    assert (
        result["execution_envelope"]["terms"]["feature_peak_days"]
        >= weekly.trader.lookback_days
    )


def test_capacity_audit_refuses_existing_output_before_source_work(tmp_path):
    audit_module = module()

    output = tmp_path / "exists.json"
    output.write_text("keep")
    args = SimpleNamespace(
        output=str(output),
        qualification_pin="missing",
        cache_reference="missing",
        weekly_config="missing",
        daily_config="missing",
    )
    with pytest.raises(ValueError, match="already exists"):
        audit_module.audit(args)
    assert output.read_text() == "keep"


def test_approved_workload_rejects_wrong_source_and_changed_scope(monkeypatch):
    audit_module = module()
    root = Path(__file__).resolve().parents[4]
    weekly = audit_module._config(
        root / "configs/hyperliquid_annual_20250901_20260901/weekly.json"
    )
    daily = audit_module._config(
        root / "configs/hyperliquid_annual_20250901_20260901/daily.json"
    )
    monkeypatch.setattr(
        audit_module,
        "file_hash",
        lambda path: (
            audit_module.APPROVED_WEEKLY_SHA256
            if str(path) == "weekly"
            else audit_module.APPROVED_DAILY_SHA256
        ),
    )
    pin = {
        "path": audit_module.APPROVED_SOURCE_PATH,
        "sha256": audit_module.APPROVED_SOURCE_SHA256,
    }

    with pytest.raises(ValueError, match="frozen approved source"):
        audit_module._approved_workload(
            pin | {"sha256": "0" * 64}, "weekly", "daily", weekly, daily
        )
    with pytest.raises(ValueError, match="scope or cadence"):
        audit_module._approved_workload(
            pin,
            "weekly",
            "daily",
            replace(weekly, end="2026-08-01"),
            daily,
        )


def test_combined_disk_peak_blocks_when_individual_components_fit(
    tmp_path, monkeypatch
):
    from .test_proxy_weekly import weekly_config

    audit_module = module()
    weekly = weekly_config(end="2026-08-08")
    daily = replace(
        weekly,
        rebalance="daily",
        trader=replace(weekly.trader, reselection="daily"),
        market_universe=replace(weekly.market_universe, reselection="daily"),
    )
    free = 1024**4
    monkeypatch.setattr(
        audit_module.shutil, "disk_usage", lambda _: SimpleNamespace(free=free)
    )
    first = audit_module._result(
        Source(), Resources(tmp_path), {"identity": "x"}, weekly, daily, {}, Counts(1)
    )
    free = first["output_disk"]["combined_required_bytes"] - 1
    result = audit_module._result(
        Source(), Resources(tmp_path), {"identity": "x"}, weekly, daily, {}, Counts(1)
    )

    assert free > result["output_disk"]["ranking_copy_bound"]
    assert "combined_filesystem_peak" in result["blocking_bounds"]
