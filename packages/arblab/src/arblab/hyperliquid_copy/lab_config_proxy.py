"""Explicit hourly proxy mode; legacy configuration payloads remain unchanged."""

from dataclasses import asdict, dataclass, field
from typing import Literal

from .lab_config_v2 import (
    ExplicitUniverse,
    FollowerSettings,
    LabConfigV2,
    LiquidityUniverse,
    TraderSettings,
)
from .proxy_simulator import ProxySimulationConfig


@dataclass(frozen=True)
class ProxySettings:
    slippage_bps: float = 5
    max_mark_age_seconds: float = 4 * 86400
    max_wait_seconds: float = 4 * 86400

    def __post_init__(self):
        checked = ProxySimulationConfig(**asdict(self))
        for name in asdict(self):
            object.__setattr__(self, name, getattr(checked, name))


@dataclass(frozen=True)
class LabConfigProxy:
    schema_version: Literal["hyperliquid_copy_lab_proxy_v1"] = (
        "hyperliquid_copy_lab_proxy_v1"
    )
    market_universe: ExplicitUniverse | LiquidityUniverse = field(
        default_factory=ExplicitUniverse
    )
    trader: TraderSettings = field(default_factory=TraderSettings)
    follower: FollowerSettings = field(
        default_factory=lambda: FollowerSettings(update_minutes=60)
    )
    proxy: ProxySettings = field(default_factory=ProxySettings)
    start: str = "2026-08-03"
    end: str = "2026-08-08"
    benchmark: Literal["btc_perp_buy_hold"] = "btc_perp_buy_hold"
    split: Literal["development"] = "development"

    def __post_init__(self):
        if self.schema_version != "hyperliquid_copy_lab_proxy_v1":
            raise ValueError("unsupported proxy configuration version")
        checked = LabConfigV2(
            **{
                k: v
                for k, v in asdict(self).items()
                if k not in ("schema_version", "proxy")
            }
        )
        for name in ("market_universe", "trader", "follower"):
            object.__setattr__(self, name, getattr(checked, name))
        if self.follower.update_minutes != 60:
            raise ValueError("proxy mode requires explicit hourly (60-minute) updates")
        object.__setattr__(
            self,
            "proxy",
            ProxySettings(**self.proxy) if isinstance(self.proxy, dict) else self.proxy,
        )
        if not isinstance(self.proxy, ProxySettings):
            raise ValueError("invalid proxy settings")

    def _strategy(self):
        return LabConfigV2(
            **{
                k: v
                for k, v in self.to_dict().items()
                if k not in ("schema_version", "proxy")
            }
        )

    def effective(self, coins, weights):
        return self._strategy().effective(coins, weights)

    def summary(self):
        return (
            self._strategy().summary()
            + f" · hourly proxy-priced · {self.proxy.slippage_bps:g}bps slippage"
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class LabConfigProxyScheduled(LabConfigProxy):
    """Separate target cadence; legacy update_minutes remains the hourly grid."""

    schema_version: Literal["hyperliquid_copy_lab_proxy_v2"] = (
        "hyperliquid_copy_lab_proxy_v2"
    )
    rebalance: Literal["daily", "weekly"] = "weekly"
    market_universe: ExplicitUniverse | LiquidityUniverse = field(
        default_factory=lambda: ExplicitUniverse(reselection="weekly")
    )
    trader: TraderSettings = field(
        default_factory=lambda: TraderSettings(reselection="weekly")
    )

    def _legacy(self):
        return LabConfigProxy(
            **{
                k: v
                for k, v in asdict(self).items()
                if k not in ("schema_version", "rebalance")
            }
        )

    def __post_init__(self):
        if (
            self.schema_version != "hyperliquid_copy_lab_proxy_v2"
            or self.rebalance not in ("daily", "weekly")
        ):
            raise ValueError("Unsupported scheduled proxy configuration")
        checked = self._legacy()
        for name in ("market_universe", "trader", "follower", "proxy"):
            object.__setattr__(self, name, getattr(checked, name))
        if (
            self.trader.reselection != self.rebalance
            or self.market_universe.reselection != self.rebalance
        ):
            raise ValueError(
                "Scheduled proxy trader/market reselection must match portfolio rebalance"
            )

    def _strategy(self):
        return self._legacy()._strategy()

    def summary(self):
        return (
            self._strategy().summary()
            + f" · {self.rebalance} trader/portfolio rebalance · hourly proxy valuation · {self.proxy.slippage_bps:g}bps slippage"
        )
