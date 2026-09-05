"""Versioned market selection composed with the established trader/copy controls."""

from dataclasses import asdict, dataclass, field
from math import isclose
from types import SimpleNamespace
from typing import Literal

from .contracts import symbol
from .lab_config import LabConfig, METRICS, bounded

AssetClass = Literal["crypto", "commodity", "equity", "index"]
CLASSES = ("crypto", "commodity", "equity", "index")
Frequency = Literal["daily", "weekly", "monthly"]


def selection_base(general, classes, reselection):
    if (
        type(general) is not bool
        or not isinstance(classes, list)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("Invalid market class selection")
    if (
        not set(classes) <= set(CLASSES)
        or (general and classes)
        or (not general and not classes)
    ):
        raise ValueError("Choose General or one or more asset classes, never both")
    if reselection not in ("daily", "weekly", "monthly"):
        raise ValueError("Invalid asset selection schedule")


@dataclass(frozen=True)
class ExplicitUniverse:
    mode: Literal["explicit"] = "explicit"
    general: bool = False
    classes: list[AssetClass] = field(default_factory=lambda: ["crypto"])
    instrument_ids: list[str] = field(default_factory=lambda: ["ETH", "SOL"])
    allocation: Literal["equal", "custom"] = "equal"
    weights: dict[str, float] | None = None
    reselection: Frequency = "daily"

    def __post_init__(self):
        selection_base(self.general, self.classes, self.reselection)
        if self.mode != "explicit" or self.general:
            raise ValueError("General cannot select explicit instruments")
        if (
            not isinstance(self.instrument_ids, list)
            or not 1 <= len(self.instrument_ids) <= 25
            or len(set(self.instrument_ids)) != len(self.instrument_ids)
        ):
            raise ValueError("Select 1–25 unique instruments")
        for instrument in self.instrument_ids:
            symbol(instrument)
        if self.allocation == "equal":
            if self.weights is not None:
                raise ValueError("Equal allocation cannot carry custom weights")
        elif self.allocation == "custom":
            if not isinstance(self.weights, dict) or set(self.weights) != set(
                self.instrument_ids
            ):
                raise ValueError("Custom budgets must match explicit instruments")
            for value in self.weights.values():
                bounded(value, "asset weight", 0.001, 1)
            if not isclose(sum(self.weights.values()), 1, abs_tol=1e-9):
                raise ValueError("Custom asset budgets must sum to one")
        else:
            raise ValueError("Unsupported allocation")
        object.__setattr__(self, "classes", sorted(self.classes))
        object.__setattr__(self, "instrument_ids", sorted(self.instrument_ids))


@dataclass(frozen=True)
class LiquidityUniverse:
    mode: Literal["liquidity"] = "liquidity"
    general: bool = True
    classes: list[AssetClass] = field(default_factory=list)
    top_n: int = 3
    min_volume_usd: float = 0
    lookback_days: int = 30
    reselection: Frequency = "daily"
    metric: Literal["traded_notional_usd"] = "traded_notional_usd"
    publication_lag_days: Literal[1] = 1

    def __post_init__(self):
        selection_base(self.general, self.classes, self.reselection)
        if self.mode != "liquidity" or self.metric != "traded_notional_usd":
            raise ValueError("Unsupported market liquidity metric")
        bounded(self.top_n, "top asset count", 1, 25, integer=True)
        bounded(self.lookback_days, "liquidity lookback", 1, 3650, integer=True)
        bounded(self.min_volume_usd, "minimum market volume", 0, 1e18)
        bounded(self.publication_lag_days, "volume publication lag", 1, 1, integer=True)
        object.__setattr__(self, "classes", sorted(self.classes))


def parse_universe(value):
    if isinstance(value, (ExplicitUniverse, LiquidityUniverse)):
        return value
    if not isinstance(value, dict):
        raise ValueError("Market universe must be an object")
    if value.get("mode") == "explicit":
        return ExplicitUniverse(**value)
    if value.get("mode") == "liquidity":
        return LiquidityUniverse(**value)
    raise ValueError("Choose explicit instruments or liquidity selection")


@dataclass(frozen=True)
class TraderSettings:
    scope: Literal["per_asset", "pooled"] = "per_asset"
    lookback_days: int = 90
    min_active_days: int = 30
    min_episodes: int = 20
    min_notional: float = 100000
    min_volume: float = 0
    min_minutes: float = 15
    metric_weights: dict[str, float] = field(
        default_factory=lambda: {k: 0.2 for k in METRICS[:5]}
    )
    metric_directions: dict[str, Literal["asc", "desc"]] = field(default_factory=dict)
    selection: Literal["fraction", "n"] = "fraction"
    top_fraction: float | None = 0.05
    top_n: int | None = None
    min_cohort: int = 5
    max_cohort: int = 25
    reselection: Frequency = "daily"


@dataclass(frozen=True)
class FollowerSettings:
    aggregation: Literal[
        "direction_equal", "direction_score_weighted", "conviction_trimmed"
    ] = "direction_equal"
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


@dataclass(frozen=True)
class LabConfigV2:
    schema_version: Literal["hyperliquid_copy_lab_v2"] = "hyperliquid_copy_lab_v2"
    market_universe: ExplicitUniverse | LiquidityUniverse = field(
        default_factory=ExplicitUniverse
    )
    trader: TraderSettings = field(default_factory=TraderSettings)
    follower: FollowerSettings = field(default_factory=FollowerSettings)
    start: str = "2026-01-05"
    end: str = "2026-01-08"
    benchmark: Literal["btc_perp_buy_hold"] = "btc_perp_buy_hold"
    split: Literal["development"] = "development"

    def __post_init__(self):
        if self.schema_version != "hyperliquid_copy_lab_v2":
            raise ValueError("Unsupported configuration version")
        object.__setattr__(
            self, "market_universe", parse_universe(self.market_universe)
        )
        trader = (
            TraderSettings(**self.trader)
            if isinstance(self.trader, dict)
            else self.trader
        )
        follower = (
            FollowerSettings(**self.follower)
            if isinstance(self.follower, dict)
            else self.follower
        )
        # Reuse the exact established scalar validation and normalization without
        # pretending arbitrary instruments are members of v1's fixed universe.
        checked = LabConfig(
            **asdict(trader),
            **asdict(follower),
            start=self.start,
            end=self.end,
            benchmark=self.benchmark,
            split=self.split,
        )
        object.__setattr__(
            self,
            "trader",
            TraderSettings(**{k: getattr(checked, k) for k in asdict(trader)}),
        )
        object.__setattr__(self, "follower", follower)

    def to_dict(self):
        return asdict(self)

    def effective(self, coins, weights):
        return SimpleNamespace(
            **asdict(self.trader),
            **asdict(self.follower),
            coins=list(coins),
            asset_weights=dict(weights),
            start=self.start,
            end=self.end,
            benchmark=self.benchmark,
            split=self.split,
        )

    def summary(self):
        u = self.market_universe
        market = (
            " + ".join(u.instrument_ids)
            if isinstance(u, ExplicitUniverse)
            else f"{'General' if u.general else '+'.join(u.classes)} top {u.top_n} assets by {u.lookback_days}d volume (1d lag)"
        )
        return f"{market} · {self.trader.scope} · {self.trader.lookback_days}d trader lookback · {self.follower.aggregation} · BTC perp benchmark"
