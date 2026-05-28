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
    bearish_3d: bool = False
    bearish_1w: bool = False


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

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}
        self.event_log: list[dict[str, Any]] = []
        self.in_full_short_mode = False
        self._last_rebalance_bar: int | None = None
        self._open_eth_short_amount = 0.0
        self._average_eth_short_basis_usdc = 0.0
        self._lifetime_realized_hedge_pnl_usdc = 0.0
        self._consumed_hedge_profit_usdc = 0.0

    def setup(
        self, snapshot: AccountSnapshot, config: Dict[str, Any]
    ) -> AccountSnapshot:
        self._config = dict(config)
        self.event_log = []
        self.in_full_short_mode = False
        self._last_rebalance_bar = None
        self._open_eth_short_amount = 0.0
        self._average_eth_short_basis_usdc = 0.0
        self._lifetime_realized_hedge_pnl_usdc = 0.0
        self._consumed_hedge_profit_usdc = 0.0

        sol_price = float(config.get("initial_sol_price", 150.0))
        eth_price = float(config.get("initial_eth_price", 2_000.0))
        initial_sol = float(config.get("initial_sol_collateral", 100.0))

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
        }

    def on_bar(
        self, snapshot: AccountSnapshot, bar: BarData
    ) -> List[Dict[str, Any]]:
        vote = self._vote(bar)
        target_ratio, reason = self._target_eth_ratio(vote)
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

        if abs(target_ratio - current_ratio) > threshold:
            if target_ratio > current_ratio:
                actions.extend(
                    self._increase_eth_short(snapshot, target_ratio - current_ratio, bar)
                )
            else:
                actions.extend(
                    self._decrease_eth_short(snapshot, current_ratio - target_ratio, bar)
                )

        if vote.green <= 1:
            actions.extend(self._defensive_usdc_repay(snapshot, vote))

        if not actions and not vote.bearish_1w:
            actions.extend(self._green_regime_usdc_debt_cleanup(snapshot))
            if actions:
                reason = "green_regime_usdc_debt_cleanup"

        if not actions and vote.green == 4 and not self.in_full_short_mode:
            actions.extend(self._bullish_relever(snapshot, bar))
            if actions:
                reason = "bullish_relever"

        if not actions and vote.green >= 3 and not self.in_full_short_mode:
            actions.extend(self._surplus_usdc_reinvestment(snapshot, vote, bar))
            if actions:
                reason = "surplus_reinvestment"

        if actions:
            self._last_rebalance_bar = bar.bar_index
            self._record_event(bar, snapshot, vote, target_ratio, reason, actions)

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
                bearish_3d=bool(raw.get("bearish_3d", False)),
                bearish_1w=bool(raw.get("bearish_1w", False)),
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

        bearish_3d = self._higher_timeframe_bearish(bar, "3d", atr_period, multiplier)
        bearish_1w = self._higher_timeframe_bearish(bar, "1w", atr_period, multiplier)
        return TrendVote(green=green, bearish_3d=bearish_3d, bearish_1w=bearish_1w)

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

    def _eth_short_ratio(self, snapshot: AccountSnapshot) -> float:
        sol_value = self._collateral_value(snapshot, "SOL")
        if sol_value <= 0:
            return 0.0
        return self._debt_value(snapshot, "ETH") / sol_value

    def _increase_eth_short(
        self,
        snapshot: AccountSnapshot,
        ratio_delta: float,
        bar: BarData,
    ) -> list[dict[str, Any]]:
        sol_value = self._collateral_value(snapshot, "SOL")
        eth_price = bar.prices["ETH"].close
        fee = self._swap_fee()
        desired_short_usd = sol_value * ratio_delta
        short_usd = min(
            desired_short_usd,
            self._max_additional_eth_short_usd(snapshot),
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

    def _max_additional_eth_short_usd(self, snapshot: AccountSnapshot) -> float:
        min_hf = float(self._config.get("min_rebalance_hf", 1.25))
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

        spend_usdc = min(
            spendable_profit * reinvest_fraction,
            sol_value
            * float(
                self._config.get("max_surplus_reinvestment_pct_of_sol_collateral", 0.05)
            ),
            self._collateral_amount(snapshot, "USDC"),
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
            - self._consumed_hedge_profit_usdc,
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
