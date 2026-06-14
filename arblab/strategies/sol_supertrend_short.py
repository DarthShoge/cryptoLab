"""SOL Supertrend strategy with ETH short hedge and full-short modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from arblab.backtest.strategy import BarData, Strategy
from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition


@dataclass(frozen=True)
class TrendVote:
    green: int
    bearish_1d: bool = False
    bearish_3d: bool = False
    bearish_1w: bool = False
    crisis_exit_1w_green: bool = False


@dataclass
class CrisisState:
    active: bool = False
    entry_reason: str | None = None
    exit_reason: str | None = None
    hedge_floor: float = 0.0
    under_hedged: bool = False
    saw_bearish_1w: bool = False
    max_sol_equiv_drawdown: float = 0.0
    partial_fill_added_usd: float = 0.0


@dataclass
class DrawdownContainmentState:
    active: bool = False
    hedge_floor: float = 0.0


@dataclass
class FastBreakState:
    active: bool = False
    hedge_floor: float = 0.0
    entered_bar: int | None = None
    exit_bar: int | None = None
    entry_reason: str | None = None
    partial_fill_added_usd: float = 0.0


@dataclass
class WeeklyBearishReserveState:
    active: bool = False
    reserve_usdc: float = 0.0
    sold_sol: float = 0.0


@dataclass
class ProfitLockReserveState:
    active: bool = False
    reserve_usdc: float = 0.0
    sold_sol: float = 0.0
    initial_slice_sold: bool = False
    escalation_slice_sold: bool = False
    last_sale_bar: int | None = None
    episode_peak_value: float = 0.0
    completed_episode_peak_value: float = 0.0


@dataclass
class CppiExposureCapState:
    active: bool = False
    protected_usdc: float = 0.0
    sold_sol: float = 0.0
    exposure_cap_usd: float = 0.0
    protected_floor_usd: float = 0.0


@dataclass
class HedgeFailureCircuitBreakerState:
    active: bool = False
    protected_usdc: float = 0.0
    sold_sol: float = 0.0
    entered_bar: int | None = None
    exit_bar: int | None = None
    relative_underperformance: float = 0.0


@dataclass
class TrafficLightGovernorState:
    active: bool = False
    green_votes: int = 4
    state: str = "green"
    hedge_floor: float = 0.0


@dataclass(frozen=True)
class HedgeTargetOverlay:
    floor: float
    up_reason: str
    down_reason: str | None = None


@dataclass
class PortfolioObservationWindow:
    max_bars: int
    portfolio_values: list[float] | None = None
    sol_equivalent_values: list[float] | None = None

    def __post_init__(self) -> None:
        self.portfolio_values = list(self.portfolio_values or [])
        self.sol_equivalent_values = list(self.sol_equivalent_values or [])

    def record(self, portfolio_value: float, sol_price: float) -> None:
        sol_equivalent = portfolio_value / sol_price if sol_price > 0 else 0.0
        self.portfolio_values.append(portfolio_value)
        self.sol_equivalent_values.append(sol_equivalent)
        self.portfolio_values = self.portfolio_values[-self.max_bars :]
        self.sol_equivalent_values = self.sol_equivalent_values[-self.max_bars :]

    def portfolio_drawdown_from_high(self) -> float:
        return self._drawdown_from_high(self.portfolio_values)

    def sol_equivalent_drawdown_from_high(self) -> float:
        return self._drawdown_from_high(self.sol_equivalent_values)

    def sol_equivalent_recovered(self, recovery_gap: float) -> bool:
        if not self.sol_equivalent_values:
            return False
        peak = max(self.sol_equivalent_values)
        if peak <= 0:
            return False
        return self.sol_equivalent_values[-1] >= peak * (1.0 - recovery_gap)

    @staticmethod
    def _drawdown_from_high(values: list[float] | None) -> float:
        if not values:
            return 0.0
        current = values[-1]
        peak = max(values)
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - current) / peak)


def supertrend_direction(
    ohlcv: pd.DataFrame,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> pd.Series:
    """Return True for bullish Supertrend state, False for bearish."""
    if ohlcv.empty:
        return pd.Series(dtype=bool)

    high = ohlcv["high"].astype(float)
    low = ohlcv["low"].astype(float)
    close = ohlcv["close"].astype(float)

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_period, min_periods=atr_period).mean()

    hl2 = (high + low) / 2.0
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    final_upper = upper_basic.copy()
    final_lower = lower_basic.copy()
    direction = pd.Series(index=ohlcv.index, dtype=bool)

    for i in range(len(ohlcv)):
        if pd.isna(atr.iloc[i]):
            direction.iloc[i] = True
            continue
        if i == 0 or pd.isna(final_upper.iloc[i - 1]) or pd.isna(final_lower.iloc[i - 1]):
            direction.iloc[i] = True
            continue

        prev_i = i - 1
        if (
            upper_basic.iloc[i] < final_upper.iloc[prev_i]
            or close.iloc[prev_i] > final_upper.iloc[prev_i]
        ):
            final_upper.iloc[i] = upper_basic.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[prev_i]

        if (
            lower_basic.iloc[i] > final_lower.iloc[prev_i]
            or close.iloc[prev_i] < final_lower.iloc[prev_i]
        ):
            final_lower.iloc[i] = lower_basic.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[prev_i]

        if direction.iloc[prev_i]:
            direction.iloc[i] = close.iloc[i] >= final_lower.iloc[i]
        else:
            direction.iloc[i] = close.iloc[i] > final_upper.iloc[i]

    return direction.fillna(True)


def resample_closed_ohlcv(
    history: pd.DataFrame,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
    """Resample symbol OHLCV and drop the in-progress final candle."""
    if symbol not in history.columns.get_level_values(0):
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    base = history[symbol]
    resampled = pd.DataFrame(
        {
            "open": base["open"].resample(timeframe).first(),
            "high": base["high"].resample(timeframe).max(),
            "low": base["low"].resample(timeframe).min(),
            "close": base["close"].resample(timeframe).last(),
            "volume": base["volume"].resample(timeframe).sum(),
        }
    ).dropna()

    if len(resampled) <= 1:
        return resampled.iloc[0:0]
    return resampled.iloc[:-1]


class SolSupertrendShortStrategy(Strategy):
    """SOL accumulation strategy using ETH short hedges on Kamino."""

    DEFAULT_HEDGE_LADDER = {
        4: 0.0,
        3: 0.25,
        2: 0.50,
        1: 0.75,
        0: 0.75,
    }
    DEFAULT_SHORT_COVER_LADDER = {
        4: 0.0,
        3: 0.35,
        2: 0.75,
        1: 1.0,
        0: None,
    }
    DRAWDOWN_CONTAINMENT_BLOCK_CONFIG = {
        "rebuy": "drawdown_containment_block_rebuy",
        "reinvestment": "drawdown_containment_block_reinvestment",
        "releverage": "drawdown_containment_block_releverage",
    }

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self.event_log: list[dict[str, Any]] = []
        self.in_full_short_mode = False
        self.crisis_state = CrisisState()
        self._last_rebalance_bar: int | None = None
        self._open_eth_short_amount = 0.0
        self._average_eth_short_basis_usdc = 0.0
        self._lifetime_realized_hedge_pnl_usdc = 0.0
        self._consumed_hedge_profit_usdc = 0.0
        self._protected_book_usdc = 0.0
        self._last_effective_hedge_target = 0.0
        self._observation_window = PortfolioObservationWindow(max_bars=1)
        self._pending_crisis_transition_reason: str | None = None
        self._initial_portfolio_value = 0.0
        self._initial_sol_equivalent = 0.0
        self._portfolio_high_watermark_value = 0.0
        self._profit_lock_active = False
        self._profit_lock_hedge_floor = 0.0
        self._profit_lock_stateful_active = False
        self.fast_break_state = FastBreakState()
        self.weekly_bearish_reserve_state = WeeklyBearishReserveState()
        self.profit_lock_reserve_state = ProfitLockReserveState()
        self.cppi_exposure_cap_state = CppiExposureCapState()
        self.hedge_failure_state = HedgeFailureCircuitBreakerState()
        self.traffic_light_governor_state = TrafficLightGovernorState()
        self._profit_lock_reserve_last_reason = "profit_lock_reserve_sell"
        self._froth_reserve_usdc = 0.0
        self._froth_reserve_executed_tiers: set[float] = set()
        self.drawdown_containment_state = DrawdownContainmentState()

    def setup(
        self, snapshot: AccountSnapshot, config: Dict[str, Any]
    ) -> AccountSnapshot:
        self._config = dict(config)
        self.event_log = []
        self.in_full_short_mode = False
        self.crisis_state = CrisisState()
        self._last_rebalance_bar = None
        self._open_eth_short_amount = 0.0
        self._average_eth_short_basis_usdc = 0.0
        self._lifetime_realized_hedge_pnl_usdc = 0.0
        self._consumed_hedge_profit_usdc = 0.0
        self._protected_book_usdc = 0.0
        self._last_effective_hedge_target = 0.0
        self._observation_window = PortfolioObservationWindow(
            max_bars=self._observation_window_bars()
        )
        self._pending_crisis_transition_reason = None
        self._profit_lock_active = False
        self._profit_lock_hedge_floor = 0.0
        self._profit_lock_stateful_active = False
        self.fast_break_state = FastBreakState()
        self.weekly_bearish_reserve_state = WeeklyBearishReserveState()
        self.profit_lock_reserve_state = ProfitLockReserveState()
        self.cppi_exposure_cap_state = CppiExposureCapState()
        self.hedge_failure_state = HedgeFailureCircuitBreakerState()
        self.traffic_light_governor_state = TrafficLightGovernorState()
        self._profit_lock_reserve_last_reason = "profit_lock_reserve_sell"
        self._froth_reserve_usdc = 0.0
        self._froth_reserve_executed_tiers = set()
        self.drawdown_containment_state = DrawdownContainmentState()

        sol_price = float(config.get("initial_sol_price", 150.0))
        eth_price = float(config.get("initial_eth_price", 2_000.0))
        initial_sol = float(config.get("initial_sol_collateral", 100.0))
        self._initial_portfolio_value = initial_sol * sol_price
        self._initial_sol_equivalent = initial_sol
        self._portfolio_high_watermark_value = self._initial_portfolio_value

        sol_ltv, sol_liq = self._asset_risk("SOL", 0.75, 0.80)
        usdc_ltv, usdc_liq = self._asset_risk("USDC", 0.90, 0.93)
        eth_borrow_factor = self._borrow_factor("ETH", 1.0)
        usdc_borrow_factor = self._borrow_factor("USDC", 1.053)

        return AccountSnapshot(
            collateral=[
                CollateralPosition("SOL", initial_sol, sol_price, sol_ltv, sol_liq),
                CollateralPosition("USDC", 0.0, 1.0, usdc_ltv, usdc_liq),
            ],
            debt=[
                DebtPosition("ETH", 0.0, eth_price, eth_borrow_factor),
                DebtPosition("USDC", 0.0, 1.0, usdc_borrow_factor),
            ],
        )

    def hedge_accounting_state(self) -> dict[str, float]:
        return {
            "open_eth_short_amount": self._open_eth_short_amount,
            "average_eth_short_basis_usdc": self._average_eth_short_basis_usdc,
            "lifetime_realized_hedge_pnl_usdc": self._lifetime_realized_hedge_pnl_usdc,
            "spendable_hedge_profit_usdc": self._spendable_hedge_profit_usdc(),
            "protected_book_usdc": self._protected_book_usdc,
        }

    def history_fields(self) -> dict[str, float | bool | str]:
        return {
            "in_crisis_mode": self.crisis_state.active,
            "crisis_hedge_floor": self.crisis_state.hedge_floor,
            "effective_hedge_target": self._last_effective_hedge_target,
            "under_hedged_crisis": self.crisis_state.under_hedged,
            "crisis_partial_fill_added_usd": self.crisis_state.partial_fill_added_usd,
            "in_profit_lock_mode": self._profit_lock_active,
            "profit_lock_hedge_floor": self._profit_lock_hedge_floor,
            "in_fast_break_overlay": self.fast_break_state.active,
            "fast_break_hedge_floor": self.fast_break_state.hedge_floor,
            "fast_break_partial_fill_added_usd": (
                self.fast_break_state.partial_fill_added_usd
            ),
            "in_weekly_bearish_reserve": self.weekly_bearish_reserve_state.active,
            "weekly_bearish_reserve_usdc": (
                self.weekly_bearish_reserve_state.reserve_usdc
            ),
            "weekly_bearish_reserve_sold_sol": (
                self.weekly_bearish_reserve_state.sold_sol
            ),
            "in_profit_lock_reserve": self.profit_lock_reserve_state.active,
            "profit_lock_reserve_usdc": self.profit_lock_reserve_state.reserve_usdc,
            "profit_lock_reserve_sold_sol": self.profit_lock_reserve_state.sold_sol,
            "in_cppi_exposure_cap": self.cppi_exposure_cap_state.active,
            "cppi_protected_usdc": self.cppi_exposure_cap_state.protected_usdc,
            "cppi_sold_sol": self.cppi_exposure_cap_state.sold_sol,
            "cppi_exposure_cap_usd": self.cppi_exposure_cap_state.exposure_cap_usd,
            "cppi_protected_floor_usd": (
                self.cppi_exposure_cap_state.protected_floor_usd
            ),
            "in_hedge_failure_circuit_breaker": self.hedge_failure_state.active,
            "hedge_failure_protected_usdc": self.hedge_failure_state.protected_usdc,
            "hedge_failure_sold_sol": self.hedge_failure_state.sold_sol,
            "hedge_failure_relative_underperformance": (
                self.hedge_failure_state.relative_underperformance
            ),
            "in_traffic_light_governor": self.traffic_light_governor_state.active,
            "traffic_light_green_votes": self.traffic_light_governor_state.green_votes,
            "traffic_light_state": self.traffic_light_governor_state.state,
            "traffic_light_hedge_floor": (
                self.traffic_light_governor_state.hedge_floor
            ),
            "protected_book_usdc": self._protected_book_usdc,
            "froth_reserve_usdc": self._froth_reserve_usdc,
            "in_drawdown_containment": self.drawdown_containment_state.active,
            "drawdown_containment_hedge_floor": (
                self.drawdown_containment_state.hedge_floor
            ),
        }

    def on_bar(
        self, snapshot: AccountSnapshot, bar: BarData
    ) -> List[Dict[str, Any]]:
        self._pending_crisis_transition_reason = None
        self._record_portfolio_observations(snapshot, bar)
        vote = self._vote(bar)
        target_ratio, reason = self._target_eth_ratio(vote)
        target_ratio, reason = self._apply_profit_lock_target(
            snapshot,
            vote,
            target_ratio,
            reason,
        )
        target_ratio, reason = self._apply_fast_break_target(
            snapshot,
            bar,
            vote,
            target_ratio,
            reason,
        )
        target_ratio, reason = self._apply_drawdown_containment_target(
            snapshot,
            vote,
            target_ratio,
            reason,
        )
        target_ratio, reason = self._apply_traffic_light_governor_target(
            snapshot,
            vote,
            target_ratio,
            reason,
        )
        target_ratio, reason = self._apply_crisis_target(
            snapshot,
            bar,
            vote,
            target_ratio,
            reason,
        )
        self._last_effective_hedge_target = target_ratio
        current_ratio = self._eth_short_ratio(snapshot)
        threshold = float(self._config.get("rebalance_threshold", 0.10))
        actions: list[dict[str, Any]] = []

        safety_action = snapshot.health_factor() < float(
            self._config.get("min_rebalance_hf", 1.25)
        )

        if (
            not safety_action
            and self._last_rebalance_bar is not None
            and bar.bar_index - self._last_rebalance_bar
            < int(self._config.get("rebalance_cooldown_bars", 4))
        ):
            self._record_event(bar, snapshot, vote, target_ratio, "cooldown_skip", [])
            return []

        if not safety_action:
            actions.extend(self._profit_lock_reserve_sell(snapshot, vote, bar))
            if actions:
                reason = self._profit_lock_reserve_last_reason

        if not actions and not safety_action:
            actions.extend(self._hedge_failure_circuit_breaker_sell(snapshot, bar))
            if actions:
                reason = "hedge_failure_circuit_breaker_sell"

        if not actions and not safety_action:
            actions.extend(self._cppi_exposure_cap_sell(snapshot, bar))
            if actions:
                reason = "cppi_exposure_cap_sell"

        if not actions and abs(target_ratio - current_ratio) > threshold:
            if self._should_fast_break_partial_fill(
                snapshot,
                target_ratio,
                current_ratio,
                reason,
                vote,
            ):
                self.crisis_state.under_hedged = self.crisis_state.active
                actions.extend(
                    self._under_hedged_fast_break_partial_fill(
                        snapshot,
                        target_ratio,
                        current_ratio,
                        bar,
                    )
                )
                if actions:
                    reason = "fast_break_partial_fill"
            elif self._is_under_hedged_crisis(snapshot, target_ratio, current_ratio):
                self.crisis_state.under_hedged = True
                actions.extend(self._under_hedged_crisis_cleanup(snapshot, bar))
                if actions:
                    if any(
                        action["type"] == "borrow" and action["symbol"] == "ETH"
                        for action in actions
                    ):
                        reason = "under_hedged_crisis_partial_fill"
                    else:
                        reason = "under_hedged_crisis_cleanup"
            elif target_ratio > current_ratio:
                self.crisis_state.under_hedged = False
                min_hf = self._hedge_add_min_hf(reason)
                actions.extend(
                    self._increase_eth_short(
                        snapshot,
                        target_ratio - current_ratio,
                        bar,
                        min_hf=min_hf,
                    )
                )
            else:
                self.crisis_state.under_hedged = False
                actions.extend(
                    self._decrease_eth_short(snapshot, current_ratio - target_ratio, bar)
                )
        elif self.crisis_state.active:
            self.crisis_state.under_hedged = False

        if vote.green <= 1:
            actions.extend(self._defensive_usdc_repay(snapshot, vote))

        if not actions and not vote.bearish_1w:
            actions.extend(self._green_regime_usdc_debt_cleanup(snapshot))
            if actions:
                reason = "green_regime_usdc_debt_cleanup"

        if not actions:
            actions.extend(self._weekly_bearish_reserve_sell(snapshot, vote, bar))
            if actions:
                reason = "weekly_bearish_reserve_sell"

        if not actions:
            actions.extend(self._weekly_bearish_reserve_rebuy(snapshot, vote, bar))
            if actions:
                reason = "weekly_bearish_reserve_rebuy"

        if not actions:
            actions.extend(self._profit_lock_reserve_rebuy(snapshot, vote, bar))
            if actions:
                reason = "profit_lock_reserve_rebuy"

        if not actions:
            actions.extend(self._cppi_exposure_cap_rebuy(snapshot, vote, bar))
            if actions:
                reason = "cppi_exposure_cap_rebuy"

        if (
            not actions
            and vote.green == 4
            and not self.in_full_short_mode
            and not self._drawdown_containment_blocks("releverage")
            and not self._traffic_light_governor_blocks_releverage(vote)
        ):
            actions.extend(self._bullish_relever(snapshot, bar))
            if actions:
                reason = "bullish_relever"

        if not actions and not self._drawdown_containment_blocks("rebuy"):
            actions.extend(self._froth_reserve_rebuy(snapshot, bar))
            if actions:
                reason = "froth_reserve_rebuy"

        if not actions:
            actions.extend(self._froth_reserve_rotation(snapshot, bar))
            if actions:
                reason = "froth_reserve_rotate"

        if (
            not actions
            and vote.green >= 3
            and not self.in_full_short_mode
            and not self.weekly_bearish_reserve_state.active
            and not self.profit_lock_reserve_state.active
            and not self.hedge_failure_state.active
            and not self._drawdown_containment_blocks("reinvestment")
            and not self._traffic_light_governor_blocks_reinvestment(vote)
        ):
            actions.extend(self._surplus_usdc_reinvestment(snapshot, vote, bar))
            if actions:
                reason = "surplus_reinvestment"

        if actions:
            self._last_rebalance_bar = bar.bar_index
            self._record_event(bar, snapshot, vote, target_ratio, reason, actions)
        elif self._pending_crisis_transition_reason is not None:
            self._record_event(
                bar,
                snapshot,
                vote,
                target_ratio,
                self._pending_crisis_transition_reason,
                [],
            )

        return actions

    def _asset_risk(
        self,
        symbol: str,
        default_ltv: float,
        default_liq: float,
    ) -> tuple[float, float]:
        market_params = self._config.get("market_params")
        cfg = getattr(market_params, "assets", {}).get(symbol) if market_params else None
        if cfg is None:
            return default_ltv, default_liq
        return cfg.ltv, cfg.liquidation_threshold

    def _borrow_factor(self, symbol: str, default: float) -> float:
        market_params = self._config.get("market_params")
        cfg = getattr(market_params, "assets", {}).get(symbol) if market_params else None
        return default if cfg is None else cfg.borrow_factor

    def _vote(self, bar: BarData) -> TrendVote:
        signal_by_bar = self._config.get("signal_by_bar")
        if signal_by_bar is not None:
            raw = signal_by_bar.get(bar.bar_index)
            if raw is None:
                return TrendVote(green=4)
            return TrendVote(
                green=int(raw.get("green", 4)),
                bearish_1d=bool(raw.get("bearish_1d", False)),
                bearish_3d=bool(raw.get("bearish_3d", False)),
                bearish_1w=bool(raw.get("bearish_1w", False)),
                crisis_exit_1w_green=bool(raw.get("crisis_exit_1w_green", False)),
            )

        atr_period = int(self._config.get("supertrend_atr_period", 10))
        multiplier = float(self._config.get("supertrend_multiplier", 3.0))
        green = 0
        for timeframe in self._config.get("timeframes", ("1h", "4h", "8h", "1d")):
            closed = resample_closed_ohlcv(bar.history, "SOL", timeframe)
            if closed.empty:
                return TrendVote(green=4)
            direction = supertrend_direction(closed, atr_period, multiplier)
            green += int(bool(direction.iloc[-1]))

        bearish_1d = self._higher_timeframe_bearish(bar, "1d", atr_period, multiplier)
        bearish_3d = self._higher_timeframe_bearish(bar, "3d", atr_period, multiplier)
        bearish_1w = self._higher_timeframe_bearish(bar, "1w", atr_period, multiplier)
        return TrendVote(
            green=green,
            bearish_1d=bearish_1d,
            bearish_3d=bearish_3d,
            bearish_1w=bearish_1w,
        )

    def _higher_timeframe_bearish(
        self,
        bar: BarData,
        timeframe: str,
        atr_period: int,
        multiplier: float,
    ) -> bool:
        closed = resample_closed_ohlcv(bar.history, "SOL", timeframe)
        if closed.empty:
            return False
        direction = supertrend_direction(closed, atr_period, multiplier)
        return not bool(direction.iloc[-1])

    def _target_eth_ratio(self, vote: TrendVote) -> tuple[float, str]:
        full_short_enabled = bool(self._config.get("enable_full_short_mode", True))
        full_short_entry = full_short_enabled and vote.green == 0 and vote.bearish_3d

        if self.in_full_short_mode or full_short_entry:
            self.in_full_short_mode = True
            if vote.green == 4:
                self.in_full_short_mode = False
                return 0.0, "full_short_cover"
            if vote.green == 0:
                lower = float(self._config.get("full_short_lower_bound", 1.0))
                upper = float(self._config.get("full_short_upper_bound", 1.5))
                target = upper if vote.bearish_1w else lower
                return target, "full_short_up"
            cover_target = self.DEFAULT_SHORT_COVER_LADDER[vote.green]
            if cover_target is None:
                cover_target = 1.0
            return float(cover_target), "full_short_cover"

        ladder = self._config.get("hedge_ladder", self.DEFAULT_HEDGE_LADDER)
        target = float(ladder.get(vote.green, 0.0))
        reason = "hedge_up" if target > 0 else "hedge_down"
        return target, reason

    def _resolve_target_overlay(
        self,
        snapshot: AccountSnapshot,
        normal_target: float,
        normal_reason: str,
        overlay: HedgeTargetOverlay,
    ) -> tuple[float, str]:
        effective_target = max(normal_target, overlay.floor)
        if effective_target <= normal_target:
            return effective_target, normal_reason
        current_ratio = self._eth_short_ratio(snapshot)
        if overlay.down_reason is not None and effective_target < current_ratio:
            return effective_target, overlay.down_reason
        return effective_target, overlay.up_reason

    def _apply_profit_lock_target(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        normal_target: float,
        normal_reason: str,
    ) -> tuple[float, str]:
        self._profit_lock_active = False
        self._profit_lock_hedge_floor = 0.0
        if not bool(self._config.get("enable_profit_lock", False)):
            self._profit_lock_stateful_active = False
            return normal_target, normal_reason
        triggered = self._profit_lock_triggered(vote)
        if bool(self._config.get("profit_lock_stateful", False)):
            if triggered:
                self._profit_lock_stateful_active = True
            elif self._profit_lock_stateful_active and self._profit_lock_stateful_exit(vote):
                self._profit_lock_stateful_active = False
            active = triggered or self._profit_lock_stateful_active
        else:
            self._profit_lock_stateful_active = False
            active = triggered
        if not active:
            return normal_target, normal_reason

        floor = float(self._config.get("profit_lock_hedge_floor", 0.35))
        self._profit_lock_active = True
        self._profit_lock_hedge_floor = floor
        return self._resolve_target_overlay(
            snapshot=snapshot,
            normal_target=normal_target,
            normal_reason=normal_reason,
            overlay=HedgeTargetOverlay(floor=floor, up_reason="profit_lock_hedge_up"),
        )

    def _profit_lock_triggered(self, vote: TrendVote) -> bool:
        trend_weak = vote.green <= int(
            self._config.get("profit_lock_max_green", 2)
        ) or vote.bearish_1d
        if not trend_weak:
            return False
        metric = str(self._config.get("profit_lock_metric", "portfolio"))
        if metric == "both":
            return self._profit_lock_metric_triggered(
                "portfolio"
            ) and self._profit_lock_metric_triggered("sol_equivalent")
        return self._profit_lock_metric_triggered(metric)

    def _profit_lock_metric_triggered(self, metric: str) -> bool:
        if metric == "sol_equivalent":
            values = self._observation_window.sol_equivalent_values
            initial = self._initial_sol_equivalent
        else:
            values = self._observation_window.portfolio_values
            initial = self._initial_portfolio_value
        if not values or initial <= 0:
            return False
        lookback = int(self._config.get("profit_lock_lookback_bars", 90 * 24))
        window = values[-lookback:]
        peak = max(window)
        current = window[-1]
        if peak <= 0:
            return False
        gain = (peak / initial) - 1.0
        drawdown = (peak - current) / peak
        if gain < float(self._config.get("profit_lock_min_gain_pct", 0.25)):
            return False
        if drawdown >= float(self._config.get("profit_lock_drawdown_threshold", 0.10)):
            return True
        near_high_threshold = float(
            self._config.get("profit_lock_near_high_threshold", 0.0)
        )
        return near_high_threshold > 0.0 and drawdown <= near_high_threshold

    def _profit_lock_stateful_exit(self, vote: TrendVote) -> bool:
        if vote.green <= int(self._config.get("profit_lock_max_green", 2)):
            return False
        if vote.bearish_1d:
            return False
        metric = str(self._config.get("profit_lock_metric", "portfolio"))
        if metric == "both":
            return self._profit_lock_metric_recovered(
                "portfolio"
            ) and self._profit_lock_metric_recovered("sol_equivalent")
        return self._profit_lock_metric_recovered(metric)

    def _profit_lock_metric_recovered(self, metric: str) -> bool:
        values = (
            self._observation_window.sol_equivalent_values
            if metric == "sol_equivalent"
            else self._observation_window.portfolio_values
        )
        if not values:
            return False
        lookback = int(self._config.get("profit_lock_lookback_bars", 90 * 24))
        window = values[-lookback:]
        peak = max(window)
        if peak <= 0:
            return False
        exit_gap = float(self._config.get("profit_lock_stateful_exit_gap", 0.02))
        return window[-1] >= peak * (1.0 - exit_gap)

    def _apply_fast_break_target(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
        vote: TrendVote,
        normal_target: float,
        normal_reason: str,
    ) -> tuple[float, str]:
        if not bool(self._config.get("enable_fast_break_overlay", False)):
            self.fast_break_state = FastBreakState()
            return normal_target, normal_reason

        if self.fast_break_state.active:
            self._update_fast_break_decay(bar)

        if self.fast_break_state.active and self._fast_break_should_exit(bar, vote):
            self.fast_break_state = FastBreakState()
            return normal_target, normal_reason

        if not self.fast_break_state.active and self._fast_break_triggered(bar, vote):
            floor = float(self._config.get("fast_break_hedge_floor", 0.75))
            hold_bars = int(self._config.get("fast_break_hold_bars", 72))
            self.fast_break_state = FastBreakState(
                active=True,
                hedge_floor=floor,
                entered_bar=bar.bar_index,
                exit_bar=bar.bar_index + hold_bars,
                entry_reason="sol_break_with_vol_expansion",
            )

        if not self.fast_break_state.active:
            return normal_target, normal_reason

        return self._resolve_target_overlay(
            snapshot=snapshot,
            normal_target=normal_target,
            normal_reason=normal_reason,
            overlay=HedgeTargetOverlay(
                floor=self.fast_break_state.hedge_floor,
                up_reason="fast_break_hedge_up",
                down_reason="fast_break_hedge_down",
            ),
        )

    def _fast_break_triggered(self, bar: BarData, vote: TrendVote) -> bool:
        max_green = int(self._config.get("fast_break_max_green", 3))
        trend_weak = vote.green <= max_green or vote.bearish_1d
        if not trend_weak:
            return False
        return self._fast_break_price_broke(bar) and self._fast_break_vol_expanded(bar)

    def _fast_break_should_exit(self, bar: BarData, vote: TrendVote) -> bool:
        if self._config.get("fast_break_decay_enabled", False):
            floors = self._fast_break_floor_schedule()
            entered = self.fast_break_state.entered_bar
            hold_bars = int(self._config.get("fast_break_hold_bars", 72))
            if entered is not None and hold_bars > 0:
                stage = max(0, (bar.bar_index - entered) // hold_bars)
                if stage >= len(floors):
                    return True
        if (
            not self._config.get("fast_break_decay_enabled", False)
            and
            self.fast_break_state.exit_bar is not None
            and bar.bar_index >= self.fast_break_state.exit_bar
        ):
            return True
        exit_green = int(self._config.get("fast_break_exit_min_green", 4))
        return vote.green >= exit_green and not vote.bearish_1d

    def _update_fast_break_decay(self, bar: BarData) -> None:
        if not bool(self._config.get("fast_break_decay_enabled", False)):
            return
        entered = self.fast_break_state.entered_bar
        hold_bars = int(self._config.get("fast_break_hold_bars", 72))
        if entered is None or hold_bars <= 0:
            return
        floors = self._fast_break_floor_schedule()
        stage = max(0, (bar.bar_index - entered) // hold_bars)
        if stage < len(floors):
            self.fast_break_state.hedge_floor = floors[stage]

    def _fast_break_floor_schedule(self) -> list[float]:
        return [
            float(self._config.get("fast_break_hedge_floor", 0.75)),
            *[
                float(floor)
                for floor in self._config.get("fast_break_decay_floors", [])
            ],
        ]

    def _fast_break_price_broke(self, bar: BarData) -> bool:
        sol_history = self._sol_close_history(bar)
        return_lookback = int(self._config.get("fast_break_return_lookback_bars", 24))
        if len(sol_history) <= return_lookback:
            return False
        lookback_return = (
            sol_history.iloc[-1] / sol_history.iloc[-return_lookback - 1] - 1.0
        )
        if lookback_return <= float(
            self._config.get("fast_break_return_threshold", -0.08)
        ):
            return True

        if not bool(self._config.get("fast_break_use_donchian_break", False)):
            return False
        donchian_lookback = int(
            self._config.get("fast_break_donchian_lookback_bars", 7 * 24)
        )
        if len(sol_history) <= donchian_lookback:
            return False
        prior_low = sol_history.iloc[-donchian_lookback - 1 : -1].min()
        return bool(sol_history.iloc[-1] < prior_low)

    def _fast_break_vol_expanded(self, bar: BarData) -> bool:
        sol_history = self._sol_close_history(bar)
        returns = sol_history.pct_change().dropna()
        vol_lookback = int(self._config.get("fast_break_vol_lookback_bars", 24))
        median_lookback = int(self._config.get("fast_break_vol_median_bars", 30 * 24))
        if len(returns) < vol_lookback * 2:
            return False
        rolling_vol = returns.rolling(vol_lookback).std().dropna()
        if rolling_vol.empty:
            return False
        current_vol = float(rolling_vol.iloc[-1])
        baseline = float(rolling_vol.tail(median_lookback).median())
        if baseline <= 0:
            return False
        return (
            current_vol
            >= float(self._config.get("fast_break_vol_multiplier", 1.5)) * baseline
        )

    @staticmethod
    def _sol_close_history(bar: BarData) -> pd.Series:
        if "SOL" not in bar.history.columns.get_level_values(0):
            return pd.Series(dtype=float)
        return bar.history["SOL"]["close"].astype(float)

    def _apply_drawdown_containment_target(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        normal_target: float,
        normal_reason: str,
    ) -> tuple[float, str]:
        if not bool(self._config.get("enable_drawdown_containment", False)):
            self.drawdown_containment_state = DrawdownContainmentState()
            return normal_target, normal_reason

        drawdown = self._portfolio_drawdown_from_rolling_high()
        trigger = float(self._config.get("drawdown_containment_trigger", 0.20))
        if not self.drawdown_containment_state.active and drawdown >= trigger:
            self.drawdown_containment_state.active = True
        elif (
            self.drawdown_containment_state.active
            and self._drawdown_containment_recovered(vote)
        ):
            self.drawdown_containment_state = DrawdownContainmentState()
            return normal_target, normal_reason

        if not self.drawdown_containment_state.active:
            self.drawdown_containment_state.hedge_floor = 0.0
            return normal_target, normal_reason

        floor = float(self._config.get("drawdown_containment_hedge_floor", 0.75))
        self.drawdown_containment_state.hedge_floor = floor
        return self._resolve_target_overlay(
            snapshot=snapshot,
            normal_target=normal_target,
            normal_reason=normal_reason,
            overlay=HedgeTargetOverlay(
                floor=floor,
                up_reason="drawdown_containment_hedge_up",
                down_reason="drawdown_containment_hedge_down",
            ),
        )

    def _drawdown_containment_recovered(self, vote: TrendVote) -> bool:
        if vote.green < 4 or vote.bearish_1d:
            return False
        values = self._observation_window.portfolio_values
        if not values:
            return False
        peak = max(values)
        if peak <= 0:
            return False
        exit_gap = float(self._config.get("drawdown_containment_exit_gap", 0.10))
        return values[-1] >= peak * (1.0 - exit_gap)

    def _drawdown_containment_blocks(self, action: str) -> bool:
        if not self.drawdown_containment_state.active:
            return False
        config_key = self.DRAWDOWN_CONTAINMENT_BLOCK_CONFIG.get(action)
        if config_key is None:
            raise ValueError(f"Unknown drawdown containment action: {action}")
        return bool(self._config.get(config_key, True))

    def _apply_traffic_light_governor_target(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        normal_target: float,
        normal_reason: str,
    ) -> tuple[float, str]:
        if not bool(self._config.get("enable_traffic_light_governor", False)):
            self.traffic_light_governor_state = TrafficLightGovernorState(
                green_votes=vote.green,
                state=self._traffic_light_state_name(vote.green),
            )
            return normal_target, normal_reason

        floor = self._traffic_light_hedge_floor(vote.green)
        self.traffic_light_governor_state = TrafficLightGovernorState(
            active=True,
            green_votes=vote.green,
            state=self._traffic_light_state_name(vote.green),
            hedge_floor=floor,
        )
        return self._resolve_target_overlay(
            snapshot=snapshot,
            normal_target=normal_target,
            normal_reason=normal_reason,
            overlay=HedgeTargetOverlay(
                floor=floor,
                up_reason="traffic_light_hedge_up",
                down_reason="traffic_light_hedge_down",
            ),
        )

    def _traffic_light_hedge_floor(self, green_votes: int) -> float:
        floors = self._config.get("traffic_light_hedge_floors", {})
        if green_votes in floors:
            return float(floors[green_votes])
        return float(floors.get(str(green_votes), 0.0))

    @staticmethod
    def _traffic_light_state_name(green_votes: int) -> str:
        return {
            4: "green",
            3: "lime",
            2: "yellow",
            1: "orange",
            0: "red",
        }.get(green_votes, "unknown")

    def _traffic_light_governor_blocks_releverage(self, vote: TrendVote) -> bool:
        if not bool(self._config.get("enable_traffic_light_governor", False)):
            return False
        min_green = int(self._config.get("traffic_light_min_releverage_green", 4))
        return vote.green < min_green

    def _traffic_light_governor_blocks_reinvestment(self, vote: TrendVote) -> bool:
        if not bool(self._config.get("enable_traffic_light_governor", False)):
            return False
        min_green = int(self._config.get("traffic_light_min_reinvestment_green", 3))
        return vote.green < min_green

    def _hedge_add_min_hf(self, reason: str) -> float | None:
        if (
            reason == "fast_break_hedge_up"
            and self._config.get("fast_break_add_min_hf") is not None
        ):
            return float(self._config.get("fast_break_add_min_hf"))
        if (
            reason == "traffic_light_hedge_up"
            and self._config.get("traffic_light_add_min_hf") is not None
        ):
            return float(self._config.get("traffic_light_add_min_hf"))
        return None

    def _apply_crisis_target(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
        vote: TrendVote,
        normal_target: float,
        normal_reason: str,
    ) -> tuple[float, str]:
        if not bool(self._config.get("enable_crisis_mode", False)):
            self.crisis_state = CrisisState()
            return normal_target, normal_reason

        just_entered_crisis = False
        if not self.crisis_state.active and self._should_enter_crisis(bar, vote):
            self.crisis_state.active = True
            self.crisis_state.entry_reason = (
                self.crisis_state.entry_reason or "sol_drawdown_with_bearish_trend"
            )
            self.crisis_state.exit_reason = None
            self.crisis_state.saw_bearish_1w = vote.bearish_1w
            self.crisis_state.partial_fill_added_usd = 0.0
            self.crisis_state.max_sol_equiv_drawdown = (
                self._sol_equivalent_drawdown_from_rolling_high()
            )
            self._pending_crisis_transition_reason = "crisis_enter"
            just_entered_crisis = True

        if not self.crisis_state.active:
            self.crisis_state.hedge_floor = 0.0
            self.crisis_state.under_hedged = False
            return normal_target, normal_reason

        if vote.bearish_1w:
            self.crisis_state.saw_bearish_1w = True
        self.crisis_state.max_sol_equiv_drawdown = max(
            self.crisis_state.max_sol_equiv_drawdown,
            self._sol_equivalent_drawdown_from_rolling_high(),
        )
        exit_reason = None if just_entered_crisis else self._crisis_exit_reason(vote)
        if exit_reason is not None:
            self.crisis_state.active = False
            self.crisis_state.exit_reason = exit_reason
            self.crisis_state.hedge_floor = 0.0
            self.crisis_state.under_hedged = False
            self.crisis_state.partial_fill_added_usd = 0.0
            self._pending_crisis_transition_reason = "crisis_exit"
            return normal_target, normal_reason

        floor = self._crisis_hedge_floor(vote)
        self.crisis_state.hedge_floor = floor
        return self._resolve_target_overlay(
            snapshot=snapshot,
            normal_target=normal_target,
            normal_reason=normal_reason,
            overlay=HedgeTargetOverlay(
                floor=floor,
                up_reason="crisis_hedge_up",
                down_reason="crisis_hedge_down",
            ),
        )

    def _should_enter_crisis(self, bar: BarData, vote: TrendVote) -> bool:
        if not self._crisis_lookback_ready(bar):
            return False
        if (
            (vote.bearish_3d or vote.bearish_1w)
            and self._sol_equivalent_drawdown_from_rolling_high()
            >= float(self._config.get("crisis_sol_equiv_drawdown_threshold", 0.25))
        ):
            self.crisis_state.entry_reason = "sol_equivalent_drawdown"
            return True
        if not vote.bearish_1d:
            return False
        sol_drawdown = self._sol_drawdown_from_rolling_high(bar)
        threshold = float(self._config.get("crisis_sol_drawdown_threshold", 0.25))
        if sol_drawdown < threshold:
            return False
        if vote.bearish_3d:
            self.crisis_state.entry_reason = "sol_drawdown_with_bearish_trend"
            return True
        if self._portfolio_drawdown_from_rolling_high() >= float(
            self._config.get("crisis_portfolio_drawdown_threshold", 0.20)
        ):
            self.crisis_state.entry_reason = "sol_drawdown_with_portfolio_damage"
            return True
        return False

    def _crisis_exit_reason(self, vote: TrendVote) -> str | None:
        if vote.crisis_exit_1w_green:
            return "weekly_recovery"
        recovery_gap = float(
            self._config.get("crisis_exit_sol_equiv_recovery_gap", 0.10)
        )
        if self._sol_equivalent_recovered(recovery_gap):
            return "sol_equivalent_recovery"
        if not bool(self._config.get("crisis_exit_on_1w_green", True)):
            return None
        if self.crisis_state.saw_bearish_1w and not vote.bearish_1w:
            return "weekly_recovery"
        return None

    def _record_portfolio_observations(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
    ) -> None:
        portfolio_value = snapshot.total_collateral_value() - snapshot.total_debt_value()
        self._portfolio_high_watermark_value = max(
            self._portfolio_high_watermark_value,
            portfolio_value,
        )
        sol_price = float(bar.prices["SOL"].close)
        self._observation_window.max_bars = self._observation_window_bars()
        self._observation_window.record(portfolio_value, sol_price)

    def _observation_window_bars(self) -> int:
        return max(
            int(self._config.get("crisis_portfolio_drawdown_lookback_bars", 30 * 24)),
            int(self._config.get("crisis_sol_equivalent_drawdown_lookback_bars", 90 * 24)),
        )

    def _portfolio_drawdown_from_rolling_high(self) -> float:
        return self._observation_window.portfolio_drawdown_from_high()

    def _sol_equivalent_drawdown_from_rolling_high(self) -> float:
        return self._observation_window.sol_equivalent_drawdown_from_high()

    def _sol_equivalent_recovered(self, recovery_gap: float) -> bool:
        if self.crisis_state.max_sol_equiv_drawdown < float(
            self._config.get("crisis_sol_equiv_drawdown_threshold", 0.25)
        ):
            return False
        return self._observation_window.sol_equivalent_recovered(recovery_gap)

    def _crisis_lookback_ready(self, bar: BarData) -> bool:
        required = max(
            int(self._config.get("crisis_sol_drawdown_lookback_bars", 60 * 24)),
            int(self._config.get("crisis_portfolio_drawdown_lookback_bars", 30 * 24)),
            int(self._config.get("crisis_sol_equivalent_drawdown_lookback_bars", 90 * 24)),
        )
        return len(bar.history) >= required

    def _sol_drawdown_from_rolling_high(self, bar: BarData) -> float:
        lookback = int(self._config.get("crisis_sol_drawdown_lookback_bars", 60 * 24))
        if "SOL" not in bar.history.columns.get_level_values(0):
            return 0.0
        sol_history = bar.history["SOL"].tail(lookback)
        if sol_history.empty:
            return 0.0
        rolling_high = float(sol_history["high"].max())
        current_price = float(bar.prices["SOL"].close)
        if rolling_high <= 0:
            return 0.0
        return max(0.0, (rolling_high - current_price) / rolling_high)

    def _crisis_hedge_floor(self, vote: TrendVote) -> float:
        if vote.bearish_3d and vote.bearish_1w:
            return float(self._config.get("crisis_hedge_floor_3d_1w", 1.25))
        if vote.bearish_3d:
            return float(self._config.get("crisis_hedge_floor_3d", 1.0))
        return float(self._config.get("crisis_hedge_floor_base", 0.75))

    def _eth_short_ratio(self, snapshot: AccountSnapshot) -> float:
        sol_value = self._collateral_value(snapshot, "SOL")
        if sol_value <= 0:
            return 0.0
        return self._debt_value(snapshot, "ETH") / sol_value

    def _is_under_hedged_crisis(
        self,
        snapshot: AccountSnapshot,
        target_ratio: float,
        current_ratio: float,
    ) -> bool:
        if not self.crisis_state.active or target_ratio <= current_ratio:
            return False
        sol_value = self._collateral_value(snapshot, "SOL")
        desired_short_usd = sol_value * (target_ratio - current_ratio)
        if desired_short_usd <= 0:
            return False
        max_short_usd = self._max_additional_eth_short_usd(snapshot)
        return max_short_usd + 1e-9 < desired_short_usd

    def _should_fast_break_partial_fill(
        self,
        snapshot: AccountSnapshot,
        target_ratio: float,
        current_ratio: float,
        reason: str,
        vote: TrendVote,
    ) -> bool:
        if reason != "fast_break_hedge_up":
            return False
        if not bool(self._config.get("enable_fast_break_partial_fill", False)):
            return False
        max_green = self._config.get("fast_break_partial_fill_max_green")
        if max_green is not None and vote.green > int(max_green):
            return False
        if (
            bool(self._config.get("fast_break_partial_fill_requires_crisis", False))
            and not self.crisis_state.active
        ):
            return False
        if target_ratio <= current_ratio:
            return False
        sol_value = self._collateral_value(snapshot, "SOL")
        desired_short_usd = sol_value * (target_ratio - current_ratio)
        if desired_short_usd <= 0:
            return False
        partial_fill_min_hf = float(
            self._config.get(
                "fast_break_partial_fill_min_hf",
                self._config.get("fast_break_add_min_hf", 2.5),
            )
        )
        budget_usd = sol_value * float(
            self._config.get("fast_break_partial_fill_budget_pct", 0.25)
        )
        remaining_budget_usd = budget_usd - self.fast_break_state.partial_fill_added_usd
        max_short_usd = min(
            self._max_additional_eth_short_usd(snapshot, partial_fill_min_hf),
            max(0.0, remaining_budget_usd),
        )
        return max_short_usd + 1e-9 < desired_short_usd

    def _under_hedged_fast_break_partial_fill(
        self,
        snapshot: AccountSnapshot,
        target_ratio: float,
        current_ratio: float,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        sol_value = self._collateral_value(snapshot, "SOL")
        if sol_value <= 0:
            return []
        partial_fill_min_hf = float(
            self._config.get(
                "fast_break_partial_fill_min_hf",
                self._config.get("fast_break_add_min_hf", 2.5),
            )
        )
        desired_short_usd = sol_value * max(0.0, target_ratio - current_ratio)
        budget_usd = sol_value * float(
            self._config.get("fast_break_partial_fill_budget_pct", 0.25)
        )
        remaining_budget_usd = max(
            0.0,
            budget_usd - self.fast_break_state.partial_fill_added_usd,
        )
        max_short_usd = min(
            desired_short_usd,
            self._max_additional_eth_short_usd(snapshot, partial_fill_min_hf),
            remaining_budget_usd,
        )
        if max_short_usd <= 0:
            return []
        actions = self._increase_eth_short(
            snapshot,
            max_short_usd / sol_value,
            bar,
            min_hf=partial_fill_min_hf,
        )
        self.fast_break_state.partial_fill_added_usd += self._borrowed_eth_usd(
            actions,
            bar,
        )
        return actions

    def _under_hedged_crisis_cleanup(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        actions = self._green_regime_usdc_debt_cleanup(snapshot)
        if actions:
            return actions

        sol_value = self._collateral_value(snapshot, "SOL")
        if sol_value <= 0:
            return []
        partial_fill_min_hf = float(
            self._config.get(
                "partial_fill_min_hf",
                max(float(self._config.get("min_rebalance_hf", 1.25)), 2.5),
            )
        )
        budget_usd = sol_value * float(
            self._config.get("crisis_partial_fill_budget_pct", 0.25)
        )
        remaining_budget_usd = max(
            0.0,
            budget_usd - self.crisis_state.partial_fill_added_usd,
        )
        max_short_usd = min(
            self._max_additional_eth_short_usd(snapshot, partial_fill_min_hf),
            remaining_budget_usd,
        )
        max_safe_ratio_delta = max_short_usd / sol_value
        if max_safe_ratio_delta > 0:
            actions = self._increase_eth_short(
                snapshot,
                max_safe_ratio_delta,
                bar,
                min_hf=partial_fill_min_hf,
            )
            self.crisis_state.partial_fill_added_usd += self._borrowed_eth_usd(
                actions,
                bar,
            )
            return actions
        return []

    def _borrowed_eth_usd(
        self,
        actions: list[dict[str, Any]],
        bar: BarData,
    ) -> float:
        eth_price = float(bar.prices["ETH"].close)
        return sum(
            float(action["amount"]) * eth_price
            for action in actions
            if action["type"] == "borrow" and action["symbol"] == "ETH"
        )

    def _increase_eth_short(
        self,
        snapshot: AccountSnapshot,
        ratio_delta: float,
        bar: BarData,
        min_hf: float | None = None,
    ) -> list[dict[str, Any]]:
        sol_value = self._collateral_value(snapshot, "SOL")
        eth_price = bar.prices["ETH"].close
        fee = self._swap_fee()
        desired_short_usd = sol_value * ratio_delta
        short_usd = min(
            desired_short_usd,
            self._max_additional_eth_short_usd(snapshot, min_hf),
        )
        if short_usd <= 0 or eth_price <= 0:
            return []
        eth_tokens = short_usd / eth_price
        usdc_proceeds = short_usd * (1.0 - fee)
        self._record_eth_short_open(eth_tokens, usdc_proceeds)
        return [
            {"type": "borrow", "symbol": "ETH", "amount": eth_tokens},
            {"type": "deposit_collateral", "symbol": "USDC", "amount": usdc_proceeds},
        ]

    def _max_additional_eth_short_usd(
        self,
        snapshot: AccountSnapshot,
        min_hf: float | None = None,
    ) -> float:
        min_hf = (
            float(self._config.get("min_rebalance_hf", 1.25))
            if min_hf is None
            else min_hf
        )
        borrow_limit = snapshot.borrow_limit()
        risk_debt = snapshot.risk_adjusted_debt_value()
        available_buffer = borrow_limit - (min_hf * risk_debt)
        if available_buffer <= 0:
            return 0.0

        usdc_ltv = self._collateral_ltv(snapshot, "USDC")
        eth_borrow_factor = self._borrow_factor("ETH", 1.0)
        collateral_credit = usdc_ltv * (1.0 - self._swap_fee())
        risk_cost = min_hf * eth_borrow_factor
        denom = risk_cost - collateral_credit
        if denom <= 0:
            return float("inf")
        return max(0.0, available_buffer / denom)

    def _decrease_eth_short(
        self,
        snapshot: AccountSnapshot,
        ratio_delta: float,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        sol_value = self._collateral_value(snapshot, "SOL")
        eth_price = bar.prices["ETH"].close
        fee = self._swap_fee()
        desired_repay_usd = sol_value * ratio_delta
        eth_debt = self._debt_amount(snapshot, "ETH")
        usdc_available = self._collateral_amount(snapshot, "USDC")
        max_repay_tokens = min(
            eth_debt,
            desired_repay_usd / eth_price if eth_price > 0 else 0.0,
            usdc_available / (eth_price * (1.0 + fee)) if eth_price > 0 else 0.0,
        )
        if max_repay_tokens <= 0:
            return []
        usdc_needed = max_repay_tokens * eth_price * (1.0 + fee)
        self._record_eth_short_cover(max_repay_tokens, usdc_needed)
        return [
            {"type": "withdraw_collateral", "symbol": "USDC", "amount": usdc_needed},
            {"type": "repay", "symbol": "ETH", "amount": max_repay_tokens},
        ]

    def _bullish_relever(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_usdc_releverage", False)):
            return []

        target_hf = float(self._config.get("target_bullish_hf", 1.35))
        equity = snapshot.total_collateral_value() - snapshot.total_debt_value()
        max_debt = equity * float(self._config.get("max_usdc_debt_to_equity", 1.0))
        current_usdc_debt = self._debt_value(snapshot, "USDC")
        budget = max(0.0, max_debt - current_usdc_debt)
        if budget <= 0:
            return []

        borrow_limit = snapshot.borrow_limit()
        risk_debt = snapshot.risk_adjusted_debt_value()
        borrow_factor = self._borrow_factor("USDC", 1.053)
        usdc_ltv = self._collateral_ltv(snapshot, "SOL")
        sol_price = bar.prices["SOL"].close
        fee = self._swap_fee()

        denom = (borrow_factor * target_hf) - usdc_ltv
        if denom <= 0:
            return []
        borrow_to_target = max(0.0, (borrow_limit - target_hf * risk_debt) / denom)
        borrow_usd = min(budget, borrow_to_target)
        if borrow_usd <= 0 or sol_price <= 0:
            return []
        sol_tokens = borrow_usd * (1.0 - fee) / sol_price
        return [
            {"type": "borrow", "symbol": "USDC", "amount": borrow_usd},
            {"type": "deposit_collateral", "symbol": "SOL", "amount": sol_tokens},
        ]

    def _surplus_usdc_reinvestment(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_surplus_usdc_reinvestment", False)):
            return []

        sol_value = self._collateral_value(snapshot, "SOL")
        sol_price = bar.prices["SOL"].close
        if sol_value <= 0 or sol_price <= 0:
            return []

        spendable_profit = self._spendable_hedge_profit_usdc()
        gate = sol_value * float(self._config.get("realized_hedge_profit_gate_pct", 0.10))
        if spendable_profit < gate:
            return []

        ladder = self._config.get("surplus_reinvestment_ladder", {3: 0.25, 4: 0.50})
        reinvest_fraction = float(ladder.get(vote.green, 0.0))
        if reinvest_fraction <= 0:
            return []

        spendable_usdc_collateral = max(
            0.0,
            self._collateral_amount(snapshot, "USDC")
            - self.cppi_exposure_cap_state.protected_usdc
            - self._protected_book_usdc,
        )
        spend_usdc = min(
            spendable_profit * reinvest_fraction,
            sol_value
            * float(
                self._config.get("max_surplus_reinvestment_pct_of_sol_collateral", 0.05)
            ),
            spendable_usdc_collateral,
        )
        spend_usdc = self._cap_spend_to_min_health_factor(snapshot, spend_usdc, sol_price)
        if spend_usdc <= 0:
            return []

        self._consumed_hedge_profit_usdc += spend_usdc
        sol_tokens = spend_usdc * (1.0 - self._swap_fee()) / sol_price
        return [
            {"type": "withdraw_collateral", "symbol": "USDC", "amount": spend_usdc},
            {"type": "deposit_collateral", "symbol": "SOL", "amount": sol_tokens},
        ]

    def _weekly_bearish_reserve_sell(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_weekly_bearish_reserve", False)):
            self.weekly_bearish_reserve_state = WeeklyBearishReserveState()
            return []
        if not vote.bearish_1w:
            self.weekly_bearish_reserve_state.active = (
                self.weekly_bearish_reserve_state.reserve_usdc > 0
            )
            return []

        sol_price = float(bar.prices["SOL"].close)
        sol_amount = self._collateral_amount(snapshot, "SOL")
        if sol_price <= 0 or sol_amount <= 0:
            return []

        self.weekly_bearish_reserve_state.active = True
        min_sol = float(
            self._config.get(
                "weekly_bearish_reserve_min_sol_collateral",
                self._config.get("initial_sol_collateral", 100.0),
            )
        )
        max_sell_by_floor = max(0.0, sol_amount - min_sol)
        episode_sol = sol_amount + self.weekly_bearish_reserve_state.sold_sol
        max_episode_sell = episode_sol * float(
            self._config.get("weekly_bearish_reserve_max_fraction", 0.30)
        )
        remaining_episode_sell = max(
            0.0,
            max_episode_sell - self.weekly_bearish_reserve_state.sold_sol,
        )
        sell_sol = min(
            sol_amount
            * float(self._config.get("weekly_bearish_reserve_sell_fraction", 0.10)),
            max_sell_by_floor,
            remaining_episode_sell,
        )
        if sell_sol <= 0:
            return []

        usdc_proceeds = sell_sol * sol_price * (1.0 - self._swap_fee())
        self.weekly_bearish_reserve_state.reserve_usdc += usdc_proceeds
        self.weekly_bearish_reserve_state.sold_sol += sell_sol
        return [
            {"type": "withdraw_collateral", "symbol": "SOL", "amount": sell_sol},
            {"type": "deposit_collateral", "symbol": "USDC", "amount": usdc_proceeds},
        ]

    def _weekly_bearish_reserve_rebuy(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_weekly_bearish_reserve", False)):
            return []
        reserve_usdc = self.weekly_bearish_reserve_state.reserve_usdc
        if reserve_usdc <= 0:
            self.weekly_bearish_reserve_state.active = False
            self.weekly_bearish_reserve_state.sold_sol = 0.0
            return []
        if vote.bearish_1w or (vote.bearish_1d and vote.bearish_3d):
            self.weekly_bearish_reserve_state.active = True
            return []

        spend_usdc = min(
            reserve_usdc
            * float(self._config.get("weekly_bearish_reserve_rebuy_fraction", 0.50)),
            reserve_usdc,
            self._collateral_amount(snapshot, "USDC"),
        )
        sol_price = float(bar.prices["SOL"].close)
        if spend_usdc <= 0 or sol_price <= 0:
            self.weekly_bearish_reserve_state.active = True
            return []

        self.weekly_bearish_reserve_state.reserve_usdc -= spend_usdc
        sol_tokens = spend_usdc * (1.0 - self._swap_fee()) / sol_price
        self.weekly_bearish_reserve_state.sold_sol = max(
            0.0,
            self.weekly_bearish_reserve_state.sold_sol - sol_tokens,
        )
        if self.weekly_bearish_reserve_state.reserve_usdc <= 1e-9:
            self.weekly_bearish_reserve_state = WeeklyBearishReserveState()
        else:
            self.weekly_bearish_reserve_state.active = True

        return [
            {"type": "withdraw_collateral", "symbol": "USDC", "amount": spend_usdc},
            {"type": "deposit_collateral", "symbol": "SOL", "amount": sol_tokens},
        ]

    def _profit_lock_reserve_sell(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_profit_lock_reserve", False)):
            self.profit_lock_reserve_state = ProfitLockReserveState()
            return []
        if not self._profit_lock_reserve_should_sell(vote):
            self.profit_lock_reserve_state.active = (
                self.profit_lock_reserve_state.reserve_usdc > 0
            )
            return []

        sol_price = float(bar.prices["SOL"].close)
        sol_amount = self._collateral_amount(snapshot, "SOL")
        if sol_price <= 0 or sol_amount <= 0:
            return []

        initial_slice = not self.profit_lock_reserve_state.initial_slice_sold
        sell_fraction = float(
            self._config.get(
                "profit_lock_reserve_sell_fraction"
                if initial_slice
                else "profit_lock_reserve_escalation_sell_fraction",
                0.10,
            )
        )
        if sell_fraction <= 0:
            return []

        min_sol = float(
            self._config.get(
                "profit_lock_reserve_min_sol_collateral",
                self._config.get("initial_sol_collateral", 100.0),
            )
        )
        max_sell_by_floor = max(0.0, sol_amount - min_sol)
        episode_sol = sol_amount + self.profit_lock_reserve_state.sold_sol
        max_episode_sell = episode_sol * float(
            self._config.get("profit_lock_reserve_max_fraction", 0.30)
        )
        remaining_episode_sell = max(
            0.0,
            max_episode_sell - self.profit_lock_reserve_state.sold_sol,
        )
        sell_sol = min(sol_amount * sell_fraction, max_sell_by_floor, remaining_episode_sell)
        if sell_sol <= 0:
            return []

        usdc_proceeds = sell_sol * sol_price * (1.0 - self._swap_fee())
        self.profit_lock_reserve_state.active = True
        self.profit_lock_reserve_state.reserve_usdc += usdc_proceeds
        self.profit_lock_reserve_state.sold_sol += sell_sol
        self.profit_lock_reserve_state.initial_slice_sold = True
        if self._profit_lock_reserve_episode_mode():
            if self.profit_lock_reserve_state.episode_peak_value <= 0:
                self.profit_lock_reserve_state.episode_peak_value = max(
                    self._observation_window.portfolio_values or [0.0]
                )
            self.profit_lock_reserve_state.last_sale_bar = bar.bar_index
            if not initial_slice:
                self.profit_lock_reserve_state.escalation_slice_sold = True
        self._profit_lock_reserve_last_reason = (
            "profit_lock_reserve_sell"
            if initial_slice
            else "profit_lock_reserve_escalate"
        )
        return [
            {"type": "withdraw_collateral", "symbol": "SOL", "amount": sell_sol},
            {"type": "deposit_collateral", "symbol": "USDC", "amount": usdc_proceeds},
        ]

    def _profit_lock_reserve_should_sell(self, vote: TrendVote) -> bool:
        if not self.profit_lock_reserve_state.initial_slice_sold:
            if (
                self._profit_lock_reserve_episode_mode()
                and not self._profit_lock_reserve_episode_can_start()
            ):
                return False
            if not self._profit_lock_active:
                return False
            if not self._profit_lock_reserve_gain_and_near_high():
                return False
            return (
                vote.bearish_1d
                or vote.green <= 2
                or self.fast_break_state.active
            )
        if (
            self._profit_lock_reserve_episode_mode()
            and self.profit_lock_reserve_state.escalation_slice_sold
        ):
            return False
        return (
            vote.bearish_3d
            or self._observation_window.portfolio_drawdown_from_high()
            >= float(self._config.get("profit_lock_reserve_escalation_drawdown", 0.15))
        )

    def _profit_lock_reserve_episode_mode(self) -> bool:
        return bool(self._config.get("profit_lock_reserve_episode_mode", False))

    def _profit_lock_reserve_episode_can_start(self) -> bool:
        completed_peak = self.profit_lock_reserve_state.completed_episode_peak_value
        if completed_peak <= 0:
            return True
        values = self._observation_window.portfolio_values
        if not values:
            return False
        reset_gap = float(
            self._config.get("profit_lock_reserve_new_high_reset_gap", 0.0)
        )
        return values[-1] > completed_peak * (1.0 + reset_gap)

    def _profit_lock_reserve_gain_and_near_high(self) -> bool:
        values = self._observation_window.portfolio_values
        if not values or self._initial_portfolio_value <= 0:
            return False
        peak = max(values)
        current = values[-1]
        if peak <= 0:
            return False
        gain = (peak / self._initial_portfolio_value) - 1.0
        drawdown = (peak - current) / peak
        return (
            gain >= float(self._config.get("profit_lock_reserve_min_gain_pct", 1.0))
            and drawdown
            <= float(
                self._config.get("profit_lock_reserve_near_high_threshold", 0.10)
            )
        )

    def _profit_lock_reserve_rebuy(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_profit_lock_reserve", False)):
            return []
        reserve_usdc = self.profit_lock_reserve_state.reserve_usdc
        if reserve_usdc <= 0:
            if self._profit_lock_reserve_episode_mode():
                self._complete_profit_lock_reserve_episode()
            else:
                self.profit_lock_reserve_state = ProfitLockReserveState()
            return []
        if vote.bearish_1w or vote.bearish_1d or vote.bearish_3d:
            self.profit_lock_reserve_state.active = True
            return []
        if (
            self._profit_lock_reserve_episode_mode()
            and not self._profit_lock_reserve_rebuy_cooldown_elapsed(bar)
        ):
            self.profit_lock_reserve_state.active = True
            return []

        spend_usdc = min(
            reserve_usdc
            * float(self._config.get("profit_lock_reserve_rebuy_fraction", 0.50)),
            reserve_usdc,
            self._collateral_amount(snapshot, "USDC"),
        )
        sol_price = float(bar.prices["SOL"].close)
        if spend_usdc <= 0 or sol_price <= 0:
            self.profit_lock_reserve_state.active = True
            return []

        self.profit_lock_reserve_state.reserve_usdc -= spend_usdc
        sol_tokens = spend_usdc * (1.0 - self._swap_fee()) / sol_price
        self.profit_lock_reserve_state.sold_sol = max(
            0.0,
            self.profit_lock_reserve_state.sold_sol - sol_tokens,
        )
        if self.profit_lock_reserve_state.reserve_usdc <= 1e-9:
            if self._profit_lock_reserve_episode_mode():
                self._complete_profit_lock_reserve_episode()
            else:
                self.profit_lock_reserve_state = ProfitLockReserveState()
        else:
            self.profit_lock_reserve_state.active = True

        return [
            {"type": "withdraw_collateral", "symbol": "USDC", "amount": spend_usdc},
            {"type": "deposit_collateral", "symbol": "SOL", "amount": sol_tokens},
        ]

    def _cppi_exposure_cap_sell(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_cppi_exposure_cap", False)):
            self.cppi_exposure_cap_state = CppiExposureCapState()
            return []

        cap_usd, floor_usd = self._cppi_exposure_budget()
        self.cppi_exposure_cap_state.exposure_cap_usd = cap_usd
        self.cppi_exposure_cap_state.protected_floor_usd = floor_usd
        if cap_usd < 0:
            return []

        sol_value = self._collateral_value(snapshot, "SOL")
        buffer_pct = float(self._config.get("cppi_exposure_buffer_pct", 0.05))
        if sol_value <= cap_usd * (1.0 + buffer_pct):
            self.cppi_exposure_cap_state.active = (
                self.cppi_exposure_cap_state.protected_usdc > 0
            )
            return []

        sol_price = float(bar.prices["SOL"].close)
        sol_amount = self._collateral_amount(snapshot, "SOL")
        if sol_price <= 0 or sol_amount <= 0:
            return []

        min_sol = float(
            self._config.get(
                "cppi_core_min_sol_collateral",
                self._config.get("initial_sol_collateral", 100.0),
            )
        )
        max_sell_by_floor = max(0.0, sol_amount - min_sol)
        max_step_sell = sol_amount * float(
            self._config.get("cppi_max_sell_fraction_per_bar", 0.10)
        )
        sell_sol = min(
            (sol_value - cap_usd) / sol_price,
            max_sell_by_floor,
            max_step_sell,
        )
        if sell_sol <= 0:
            return []

        usdc_proceeds = sell_sol * sol_price * (1.0 - self._swap_fee())
        self.cppi_exposure_cap_state.active = True
        self.cppi_exposure_cap_state.protected_usdc += usdc_proceeds
        self.cppi_exposure_cap_state.sold_sol += sell_sol
        return [
            {"type": "withdraw_collateral", "symbol": "SOL", "amount": sell_sol},
            {"type": "deposit_collateral", "symbol": "USDC", "amount": usdc_proceeds},
        ]

    def _hedge_failure_circuit_breaker_sell(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_hedge_failure_circuit_breaker", False)):
            self.hedge_failure_state = HedgeFailureCircuitBreakerState()
            return []

        if self.hedge_failure_state.active and self._hedge_failure_should_exit(bar):
            protected_usdc = self.hedge_failure_state.protected_usdc
            sold_sol = self.hedge_failure_state.sold_sol
            self.hedge_failure_state = HedgeFailureCircuitBreakerState(
                protected_usdc=protected_usdc,
                sold_sol=sold_sol,
            )
            return []

        underperformance = self._sol_eth_relative_underperformance(bar)
        self.hedge_failure_state.relative_underperformance = underperformance
        if not self.hedge_failure_state.active:
            if not self._hedge_failure_triggered(underperformance):
                return []
            hold_bars = int(self._config.get("hedge_failure_hold_bars", 168))
            self.hedge_failure_state.active = True
            self.hedge_failure_state.entered_bar = bar.bar_index
            self.hedge_failure_state.exit_bar = bar.bar_index + hold_bars

        sell_fraction = float(self._config.get("hedge_failure_sell_fraction", 0.0))
        if sell_fraction <= 0 or self.hedge_failure_state.sold_sol > 0:
            return []

        sol_price = float(bar.prices["SOL"].close)
        sol_amount = self._collateral_amount(snapshot, "SOL")
        if sol_price <= 0 or sol_amount <= 0:
            return []
        min_sol = float(
            self._config.get(
                "hedge_failure_min_sol_collateral",
                self._config.get("initial_sol_collateral", 100.0),
            )
        )
        max_sell_by_floor = max(0.0, sol_amount - min_sol)
        sell_sol = min(sol_amount * sell_fraction, max_sell_by_floor)
        if sell_sol <= 0:
            return []
        usdc_proceeds = sell_sol * sol_price * (1.0 - self._swap_fee())
        self.hedge_failure_state.protected_usdc += usdc_proceeds
        self.hedge_failure_state.sold_sol += sell_sol
        return [
            {"type": "withdraw_collateral", "symbol": "SOL", "amount": sell_sol},
            {"type": "deposit_collateral", "symbol": "USDC", "amount": usdc_proceeds},
        ]

    def _hedge_failure_triggered(self, underperformance: float) -> bool:
        if not (
            self.fast_break_state.active
            or self._profit_lock_active
            or self.crisis_state.active
            or self.drawdown_containment_state.active
        ):
            return False
        return underperformance >= float(
            self._config.get("hedge_failure_underperformance_threshold", 0.10)
        )

    def _hedge_failure_should_exit(self, bar: BarData) -> bool:
        exit_bar = self.hedge_failure_state.exit_bar
        return exit_bar is not None and bar.bar_index >= exit_bar

    def _sol_eth_relative_underperformance(self, bar: BarData) -> float:
        lookback = int(self._config.get("hedge_failure_lookback_bars", 72))
        history = bar.history
        if len(history) <= lookback:
            return 0.0
        if not {"SOL", "ETH"}.issubset(set(history.columns.get_level_values(0))):
            return 0.0
        sol_close = history["SOL"]["close"].astype(float)
        eth_close = history["ETH"]["close"].astype(float)
        sol_return = (sol_close.iloc[-1] / sol_close.iloc[-lookback - 1]) - 1.0
        eth_return = (eth_close.iloc[-1] / eth_close.iloc[-lookback - 1]) - 1.0
        return eth_return - sol_return

    def _cppi_exposure_cap_rebuy(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_cppi_exposure_cap", False)):
            return []
        protected_usdc = self.cppi_exposure_cap_state.protected_usdc
        if protected_usdc <= 0:
            self.cppi_exposure_cap_state.active = False
            self.cppi_exposure_cap_state.sold_sol = 0.0
            return []
        if not self._cppi_rebuy_trend_confirmed(vote):
            self.cppi_exposure_cap_state.active = True
            return []

        cap_usd, floor_usd = self._cppi_exposure_budget()
        self.cppi_exposure_cap_state.exposure_cap_usd = cap_usd
        self.cppi_exposure_cap_state.protected_floor_usd = floor_usd
        sol_value = self._collateral_value(snapshot, "SOL")
        buffer_pct = float(self._config.get("cppi_exposure_buffer_pct", 0.05))
        gap_usd = cap_usd - sol_value * (1.0 + buffer_pct)
        if gap_usd <= 0:
            self.cppi_exposure_cap_state.active = True
            return []

        spend_usdc = min(
            gap_usd * float(self._config.get("cppi_rebuy_fraction", 0.50)),
            protected_usdc,
            self._collateral_amount(snapshot, "USDC"),
        )
        sol_price = float(bar.prices["SOL"].close)
        if spend_usdc <= 0 or sol_price <= 0:
            self.cppi_exposure_cap_state.active = True
            return []

        self.cppi_exposure_cap_state.protected_usdc -= spend_usdc
        sol_tokens = spend_usdc * (1.0 - self._swap_fee()) / sol_price
        self.cppi_exposure_cap_state.sold_sol = max(
            0.0,
            self.cppi_exposure_cap_state.sold_sol - sol_tokens,
        )
        if self.cppi_exposure_cap_state.protected_usdc <= 1e-9:
            self.cppi_exposure_cap_state = CppiExposureCapState(
                exposure_cap_usd=cap_usd,
                protected_floor_usd=floor_usd,
            )
        else:
            self.cppi_exposure_cap_state.active = True

        return [
            {"type": "withdraw_collateral", "symbol": "USDC", "amount": spend_usdc},
            {"type": "deposit_collateral", "symbol": "SOL", "amount": sol_tokens},
        ]

    def _cppi_exposure_budget(self) -> tuple[float, float]:
        if self._initial_portfolio_value <= 0:
            return -1.0, 0.0
        high_watermark = self._portfolio_high_watermark_value
        activation_gain = float(self._config.get("cppi_activation_gain", 1.5))
        if high_watermark < self._initial_portfolio_value * activation_gain:
            return -1.0, 0.0
        floor_usd = high_watermark * float(
            self._config.get("cppi_protect_pct", 0.65)
        )
        current = (
            self._observation_window.portfolio_values[-1]
            if self._observation_window.portfolio_values
            else high_watermark
        )
        cushion = max(0.0, current - floor_usd)
        cap_usd = cushion * float(self._config.get("cppi_cushion_multiplier", 2.0))
        return cap_usd, floor_usd

    def _cppi_rebuy_trend_confirmed(self, vote: TrendVote) -> bool:
        if vote.bearish_1w or vote.bearish_3d or vote.bearish_1d:
            return False
        return vote.green >= int(self._config.get("cppi_rebuy_min_green", 4))

    def _profit_lock_reserve_rebuy_cooldown_elapsed(self, bar: BarData) -> bool:
        last_sale_bar = self.profit_lock_reserve_state.last_sale_bar
        if last_sale_bar is None:
            return True
        cooldown_bars = int(
            self._config.get("profit_lock_reserve_rebuy_cooldown_bars", 0)
        )
        return bar.bar_index >= last_sale_bar + cooldown_bars

    def _complete_profit_lock_reserve_episode(self) -> None:
        completed_peak = max(
            self.profit_lock_reserve_state.completed_episode_peak_value,
            self.profit_lock_reserve_state.episode_peak_value,
        )
        self.profit_lock_reserve_state = ProfitLockReserveState(
            completed_episode_peak_value=completed_peak,
        )

    def _froth_reserve_rotation(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_froth_reserve", False)):
            return []
        if not self._profit_lock_active:
            return []
        sol_price = float(bar.prices["SOL"].close)
        if sol_price <= 0:
            return []
        initial = self._initial_portfolio_value
        portfolio_value = snapshot.total_collateral_value() - snapshot.total_debt_value()
        if initial <= 0:
            return []
        gain_multiple = (portfolio_value / initial) - 1.0
        tiers = self._config.get("froth_reserve_tiers", {})
        eligible = sorted(
            float(tier)
            for tier in tiers
            if gain_multiple >= float(tier)
            and float(tier) not in self._froth_reserve_executed_tiers
        )
        if not eligible:
            return []
        tier = eligible[0]
        rotate_fraction = float(tiers[tier])
        if rotate_fraction <= 0:
            self._froth_reserve_executed_tiers.add(tier)
            return []
        sol_amount = self._collateral_amount(snapshot, "SOL")
        min_sol = float(
            self._config.get(
                "froth_reserve_min_sol_collateral",
                self._config.get("initial_sol_collateral", 100.0),
            )
        )
        max_sell_sol = max(0.0, sol_amount - min_sol)
        sell_sol = min(sol_amount * rotate_fraction, max_sell_sol)
        if sell_sol <= 0:
            return []
        usdc_proceeds = sell_sol * sol_price * (1.0 - self._swap_fee())
        self._froth_reserve_usdc += usdc_proceeds
        self._froth_reserve_executed_tiers.add(tier)
        return [
            {"type": "withdraw_collateral", "symbol": "SOL", "amount": sell_sol},
            {"type": "deposit_collateral", "symbol": "USDC", "amount": usdc_proceeds},
        ]

    def _froth_reserve_rebuy(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        if not bool(self._config.get("enable_froth_reserve", False)):
            return []
        if self._froth_reserve_usdc <= 0:
            return []
        drawdown = self._sol_drawdown_from_rolling_high(bar)
        threshold = float(
            self._config.get("froth_reserve_rebuy_drawdown_threshold", 0.35)
        )
        if drawdown < threshold:
            return []
        usdc_available = self._collateral_amount(snapshot, "USDC")
        spend_usdc = min(
            self._froth_reserve_usdc
            * float(self._config.get("froth_reserve_rebuy_fraction", 0.25)),
            self._froth_reserve_usdc,
            usdc_available,
        )
        sol_price = float(bar.prices["SOL"].close)
        if spend_usdc <= 0 or sol_price <= 0:
            return []
        self._froth_reserve_usdc -= spend_usdc
        sol_tokens = spend_usdc * (1.0 - self._swap_fee()) / sol_price
        return [
            {"type": "withdraw_collateral", "symbol": "USDC", "amount": spend_usdc},
            {"type": "deposit_collateral", "symbol": "SOL", "amount": sol_tokens},
        ]

    def _cap_spend_to_min_health_factor(
        self,
        snapshot: AccountSnapshot,
        desired_spend_usdc: float,
        sol_price: float,
    ) -> float:
        min_hf = float(self._config.get("surplus_reinvestment_min_hf", 2.0))
        if desired_spend_usdc <= 0 or snapshot.risk_adjusted_debt_value() <= 0:
            return desired_spend_usdc

        usdc_ltv = self._collateral_ltv(snapshot, "USDC")
        sol_ltv = self._collateral_ltv(snapshot, "SOL")
        credit_loss_per_usdc = usdc_ltv - (sol_ltv * (1.0 - self._swap_fee()))
        if credit_loss_per_usdc <= 0:
            return desired_spend_usdc

        current_buffer = snapshot.borrow_limit() - (
            min_hf * snapshot.risk_adjusted_debt_value()
        )
        if current_buffer <= 0:
            return 0.0
        max_safe_spend = current_buffer / credit_loss_per_usdc
        return min(desired_spend_usdc, max_safe_spend)

    def _defensive_usdc_repay(
        self,
        snapshot: AccountSnapshot,
        vote: TrendVote,
    ) -> list[dict[str, Any]]:
        targets = self._config.get("defensive_usdc_debt_targets", {1: 0.50, 0: 0.0})
        target_fraction = float(targets.get(vote.green, 0.0))
        equity = snapshot.total_collateral_value() - snapshot.total_debt_value()
        budget = max(0.0, equity) * float(
            self._config.get("max_usdc_debt_to_equity", 1.0)
        )
        target_debt = budget * target_fraction
        current_debt = self._debt_amount(snapshot, "USDC")
        repay_amount = max(0.0, current_debt - target_debt)
        usdc_available = self._collateral_amount(snapshot, "USDC")
        repay_amount = min(repay_amount, usdc_available, current_debt)
        if repay_amount <= 0:
            return []
        return [
            {"type": "withdraw_collateral", "symbol": "USDC", "amount": repay_amount},
            {"type": "repay", "symbol": "USDC", "amount": repay_amount},
        ]

    def _green_regime_usdc_debt_cleanup(
        self,
        snapshot: AccountSnapshot,
    ) -> list[dict[str, Any]]:
        current_debt = self._debt_amount(snapshot, "USDC")
        usdc_available = self._collateral_amount(snapshot, "USDC")
        repay_amount = min(current_debt, usdc_available)
        if repay_amount <= 0:
            return []
        return [
            {"type": "withdraw_collateral", "symbol": "USDC", "amount": repay_amount},
            {"type": "repay", "symbol": "USDC", "amount": repay_amount},
        ]

    def _record_event(
        self,
        bar: BarData,
        snapshot: AccountSnapshot,
        vote: TrendVote,
        target_ratio: float,
        reason: str,
        actions: list[dict[str, Any]],
    ) -> None:
        self.event_log.append(
            {
                "timestamp": bar.timestamp,
                "bar_index": bar.bar_index,
                "green_votes": vote.green,
                "red_votes": 4 - vote.green,
                "bearish_3d": vote.bearish_3d,
                "bearish_1w": vote.bearish_1w,
                "target_eth_short_ratio": target_ratio,
                "in_crisis_mode": self.crisis_state.active,
                "crisis_entry_reason": self.crisis_state.entry_reason,
                "crisis_exit_reason": self.crisis_state.exit_reason,
                "crisis_hedge_floor": self.crisis_state.hedge_floor,
                "under_hedged_crisis": self.crisis_state.under_hedged,
                "crisis_partial_fill_added_usd": (
                    self.crisis_state.partial_fill_added_usd
                ),
                "in_profit_lock_mode": self._profit_lock_active,
                "profit_lock_hedge_floor": self._profit_lock_hedge_floor,
                "in_fast_break_overlay": self.fast_break_state.active,
                "fast_break_hedge_floor": self.fast_break_state.hedge_floor,
                "fast_break_partial_fill_added_usd": (
                    self.fast_break_state.partial_fill_added_usd
                ),
                "in_weekly_bearish_reserve": (
                    self.weekly_bearish_reserve_state.active
                ),
                "weekly_bearish_reserve_usdc": (
                    self.weekly_bearish_reserve_state.reserve_usdc
                ),
                "weekly_bearish_reserve_sold_sol": (
                    self.weekly_bearish_reserve_state.sold_sol
                ),
                "in_profit_lock_reserve": self.profit_lock_reserve_state.active,
                "profit_lock_reserve_usdc": (
                    self.profit_lock_reserve_state.reserve_usdc
                ),
                "profit_lock_reserve_sold_sol": (
                    self.profit_lock_reserve_state.sold_sol
                ),
                "in_cppi_exposure_cap": self.cppi_exposure_cap_state.active,
                "cppi_protected_usdc": self.cppi_exposure_cap_state.protected_usdc,
                "cppi_sold_sol": self.cppi_exposure_cap_state.sold_sol,
                "cppi_exposure_cap_usd": (
                    self.cppi_exposure_cap_state.exposure_cap_usd
                ),
                "cppi_protected_floor_usd": (
                    self.cppi_exposure_cap_state.protected_floor_usd
                ),
                "in_hedge_failure_circuit_breaker": self.hedge_failure_state.active,
                "hedge_failure_protected_usdc": self.hedge_failure_state.protected_usdc,
                "hedge_failure_sold_sol": self.hedge_failure_state.sold_sol,
                "hedge_failure_relative_underperformance": (
                    self.hedge_failure_state.relative_underperformance
                ),
                "in_traffic_light_governor": (
                    self.traffic_light_governor_state.active
                ),
                "traffic_light_green_votes": (
                    self.traffic_light_governor_state.green_votes
                ),
                "traffic_light_state": self.traffic_light_governor_state.state,
                "traffic_light_hedge_floor": (
                    self.traffic_light_governor_state.hedge_floor
                ),
                "froth_reserve_usdc": self._froth_reserve_usdc,
                "in_drawdown_containment": self.drawdown_containment_state.active,
                "drawdown_containment_hedge_floor": (
                    self.drawdown_containment_state.hedge_floor
                ),
                "current_eth_short_ratio": self._eth_short_ratio(snapshot),
                "health_factor": snapshot.health_factor(),
                "reason": reason,
                "actions": actions,
                **self.hedge_accounting_state(),
            }
        )

    def _swap_fee(self) -> float:
        return float(self._config.get("swap_fee_bps", 10.0)) / 10_000.0

    def _record_eth_short_open(self, eth_amount: float, usdc_proceeds: float) -> None:
        if eth_amount <= 0:
            return
        existing_proceeds_basis = (
            self._open_eth_short_amount * self._average_eth_short_basis_usdc
        )
        new_total_amount = self._open_eth_short_amount + eth_amount
        self._average_eth_short_basis_usdc = (
            existing_proceeds_basis + usdc_proceeds
        ) / new_total_amount
        self._open_eth_short_amount = new_total_amount

    def _record_eth_short_cover(self, eth_amount: float, usdc_cost: float) -> None:
        if eth_amount <= 0 or self._open_eth_short_amount <= 0:
            return
        covered_amount = min(eth_amount, self._open_eth_short_amount)
        cover_cost_basis = usdc_cost * (covered_amount / eth_amount)
        realized_pnl = (
            covered_amount * self._average_eth_short_basis_usdc
        ) - cover_cost_basis
        self._lifetime_realized_hedge_pnl_usdc += realized_pnl
        if bool(self._config.get("enable_protected_book", False)) and realized_pnl > 0:
            self._protected_book_usdc += realized_pnl * float(
                self._config.get("protected_book_realized_pnl_fraction", 0.25)
            )
        self._open_eth_short_amount = max(
            0.0,
            self._open_eth_short_amount - covered_amount,
        )
        if self._open_eth_short_amount == 0.0:
            self._average_eth_short_basis_usdc = 0.0

    def _spendable_hedge_profit_usdc(self) -> float:
        return max(
            0.0,
            self._lifetime_realized_hedge_pnl_usdc
            - self._consumed_hedge_profit_usdc
            - self._protected_book_usdc,
        )

    @staticmethod
    def _collateral_amount(snapshot: AccountSnapshot, symbol: str) -> float:
        pos = next((p for p in snapshot.collateral if p.symbol == symbol), None)
        return 0.0 if pos is None else pos.amount

    @staticmethod
    def _collateral_value(snapshot: AccountSnapshot, symbol: str) -> float:
        pos = next((p for p in snapshot.collateral if p.symbol == symbol), None)
        return 0.0 if pos is None else pos.value()

    @staticmethod
    def _collateral_ltv(snapshot: AccountSnapshot, symbol: str) -> float:
        pos = next((p for p in snapshot.collateral if p.symbol == symbol), None)
        return 0.0 if pos is None else pos.ltv

    @staticmethod
    def _debt_amount(snapshot: AccountSnapshot, symbol: str) -> float:
        pos = next((p for p in snapshot.debt if p.symbol == symbol), None)
        return 0.0 if pos is None else pos.amount

    @staticmethod
    def _debt_value(snapshot: AccountSnapshot, symbol: str) -> float:
        pos = next((p for p in snapshot.debt if p.symbol == symbol), None)
        return 0.0 if pos is None else pos.value()
