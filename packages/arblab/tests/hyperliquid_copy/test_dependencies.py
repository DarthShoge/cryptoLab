"""Qualify the external packages before relying on their data contracts."""

from importlib.metadata import PackageNotFoundError, version

import pytest


@pytest.mark.parametrize(
    ("distribution", "expected"),
    [("hyperliquid-data", "0.1.0"), ("hyperliquid-python-sdk", "0.24.0"), ("duckdb", "1.5.5")],
)
def test_required_distribution(distribution, expected):
    try:
        actual = version(distribution)
    except PackageNotFoundError:
        pytest.fail(f"missing required distribution: {distribution}")
    assert actual == expected


def test_fill_schema_preservation_requirement():
    from hyperliquid_data import FillRow

    assert set(FillRow.__annotations__) == {
        "time", "coin", "px", "sz", "side", "crossed", "tid",
        "user", "closed_pnl", "liquidation",
    }
