"""Versioned effective copy-strategy configuration, independent of HTTP/UI."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from math import isfinite, isclose
from typing import Literal

METRICS = (
    "pnl_efficiency",
    "profit_factor",
    "positive_day_rate",
    "drawdown_efficiency",
    "copyability",
    "gross_volume",
)


def bounded(value, name, low, high, *, integer=False):
    if (
        type(value) not in ((int,) if integer else (int, float))
        or not isfinite(value)
        or not low <= value <= high
    ):
        raise ValueError(
            f"{name} must be {'an integer' if integer else 'a number'} between {low} and {high}"
        )


def day(value):
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("Expected a UTC date in YYYY-MM-DD format")
    return datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class LabConfig:
    schema_version: Literal["hyperliquid_copy_lab_v1"] = "hyperliquid_copy_lab_v1"
    coins: list[str] = field(default_factory=lambda: ["ETH", "SOL"])
    scope: Literal["per_asset", "pooled"] = "per_asset"
    lookback_days: int = 90
    min_active_days: int = 30
    min_episodes: int = 20
    min_notional: float = 100000
    min_volume: float = 0
    min_minutes: float = 15
    metric_weights: dict[str, float] = field(
        default_factory=lambda: {key: 0.2 for key in METRICS[:5]}
    )
    metric_directions: dict[str, Literal["asc", "desc"]] = field(default_factory=dict)
    selection: Literal["fraction", "n"] = "fraction"
    top_fraction: float | None = 0.05
    top_n: int | None = None
    min_cohort: int = 5
    max_cohort: int = 25
    reselection: Literal["daily", "weekly", "monthly"] = "daily"
    aggregation: Literal[
        "direction_equal", "direction_score_weighted", "conviction_trimmed"
    ] = "direction_equal"
    asset_weights: dict[str, float] = field(
        default_factory=lambda: {"ETH": 0.5, "SOL": 0.5}
    )
    initial_equity: float = 10000
    gross_cap: float = 1
    asset_cap: float = 0.5
    update_minutes: int = 1
    latency_seconds: int = 5
    fee_bps: float = 4.5
    deadband: float = 0.02
    min_trade_usd: float = 10
    min_known: int = 5
    min_known_weight: float = 0.6
    scale_lookback_days: int = 30
    scale_quantile: float = 0.95
    trim: float = 0.1
    start: str = "2026-01-05"
    end: str = "2026-01-08"
    benchmark: Literal["btc_perp_buy_hold"] = "btc_perp_buy_hold"
    split: Literal["development"] = "development"

    def __post_init__(self):
        choices = dict(
            schema_version={"hyperliquid_copy_lab_v1"},
            scope={"per_asset", "pooled"},
            selection={"fraction", "n"},
            reselection={"daily", "weekly", "monthly"},
            aggregation={
                "direction_equal",
                "direction_score_weighted",
                "conviction_trimmed",
            },
            benchmark={"btc_perp_buy_hold"},
            split={"development"},
        )
        for key, allowed in choices.items():
            if getattr(self, key) not in allowed:
                raise ValueError(f"Unsupported {key}")
        if (
            not isinstance(self.coins, list)
            or not self.coins
            or len(set(self.coins)) != len(self.coins)
            or not set(self.coins) <= {"BTC", "ETH", "SOL"}
        ):
            raise ValueError("Choose unique copied markets from BTC, ETH and SOL")
        for key, low, high in (
            ("lookback_days", 1, 3650),
            ("min_active_days", 0, 3650),
            ("min_episodes", 0, 100000),
            ("min_cohort", 1, 250),
            ("max_cohort", 1, 250),
            ("update_minutes", 1, 1440),
            ("latency_seconds", 0, 60),
            ("min_known", 1, 250),
            ("scale_lookback_days", 1, 3650),
        ):
            bounded(getattr(self, key), key, low, high, integer=True)
        for key, low, high in (
            ("min_notional", 0, 1e15),
            ("min_volume", 0, 1e15),
            ("min_minutes", 0, 1e7),
            ("initial_equity", 1, 1e9),
            ("gross_cap", 0.001, 1),
            ("asset_cap", 0.001, 1),
            ("fee_bps", 0, 100),
            ("deadband", 0, 1),
            ("min_trade_usd", 0, 1e9),
            ("min_known_weight", 0.001, 1),
            ("scale_quantile", 0.001, 1),
            ("trim", 0, 0.49),
        ):
            bounded(getattr(self, key), key, low, high)
        if self.max_cohort < self.min_cohort:
            raise ValueError("Maximum cohort must be at least minimum cohort")
        if self.selection == "fraction":
            bounded(self.top_fraction, "top_fraction", 0.0001, 1)
            if self.top_n is not None:
                raise ValueError("Top N must be absent in fraction mode")
        else:
            bounded(self.top_n, "top_n", 1, 250, integer=True)
            if self.top_fraction is not None:
                raise ValueError("Top fraction must be absent in N mode")
        if (
            not isinstance(self.metric_weights, dict)
            or not self.metric_weights
            or not set(self.metric_weights) <= set(METRICS)
        ):
            raise ValueError("Unsupported ranking metric; wallet ROI is not available")
        for key, value in self.metric_weights.items():
            bounded(value, key, 0, 1e6)
        total = sum(self.metric_weights.values())
        if total <= 0:
            raise ValueError("At least one ranking weight must be positive")
        if (
            not isinstance(self.metric_directions, dict)
            or not set(self.metric_directions) <= set(self.metric_weights)
            or any(v not in {"asc", "desc"} for v in self.metric_directions.values())
        ):
            raise ValueError("Ranking directions must refer to selected metrics")
        if not isinstance(self.asset_weights, dict) or set(self.asset_weights) != set(
            self.coins
        ):
            raise ValueError("Asset budgets must match copied markets")
        for key, value in self.asset_weights.items():
            bounded(value, key, 0.001, 1)
        if not isclose(sum(self.asset_weights.values()), 1, abs_tol=1e-9):
            raise ValueError("Asset budgets must sum to one")
        if day(self.end) <= day(self.start):
            raise ValueError("End date must follow start date")
        object.__setattr__(self, "coins", sorted(self.coins))
        object.__setattr__(
            self,
            "metric_weights",
            {k: v / total for k, v in sorted(self.metric_weights.items()) if v > 0},
        )
        object.__setattr__(
            self,
            "metric_directions",
            {k: self.metric_directions.get(k, "desc") for k in self.metric_weights},
        )

    @classmethod
    def from_dict(cls, values):
        return cls(**values)

    def to_dict(self):
        return asdict(self)

    def summary(self):
        selection = (
            f"top {self.top_fraction * 100:g}%"
            if self.selection == "fraction"
            else f"top {self.top_n}"
        )
        return f"{' + '.join(self.coins)} · {self.scope.replace('_', ' ')} · {selection} eligible · {self.lookback_days}d lookback · {self.reselection} · {self.aggregation.replace('_', ' ')} · BTC perp benchmark"
