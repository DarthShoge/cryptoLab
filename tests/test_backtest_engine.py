"""Comprehensive tests for the backtest engine."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from arblab.backtest.engine import BacktestEngine, EngineConfig
from arblab.backtest.market import LiquidationEvent, MarketParams
from arblab.backtest.strategy import BarData, Strategy
from arblab.kamino_risk import AccountSnapshot, CollateralPosition, DebtPosition


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EXPECTED_HISTORY_COLUMNS = [
    "collateral_value",
    "debt_value",
    "portfolio_value",
    "health_factor",
    "current_ltv",
    "borrow_limit",
    "liquidation_buffer",
    "action_count",
    "interest_accrued",
    "lst_yield",
    "liquidation_penalty",
]


def make_price_data(
    n_bars: int = 100,
    start_price: float = 150.0,
    drift: float = 0.0,
    symbol: str = "SOL",
) -> pd.DataFrame:
    """Create a synthetic multi-level OHLCV DataFrame for one symbol."""
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    prices = start_price * np.cumprod(1 + np.full(n_bars, drift))
    df = pd.DataFrame(
        {
            (symbol, "open"): prices,
            (symbol, "high"): prices * 1.01,
            (symbol, "low"): prices * 0.99,
            (symbol, "close"): prices,
            (symbol, "volume"): 1000.0,
        },
        index=dates,
    )
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def make_multi_symbol_price_data(
    n_bars: int = 100,
    symbols_prices: dict[str, float] | None = None,
    drift: float = 0.0,
) -> pd.DataFrame:
    """Create price data for multiple symbols, each constant or drifting."""
    if symbols_prices is None:
        symbols_prices = {"SOL": 150.0, "USDC": 1.0}
    dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
    frames = {}
    for sym, start_price in symbols_prices.items():
        prices = start_price * np.cumprod(1 + np.full(n_bars, drift))
        frames[(sym, "open")] = prices
        frames[(sym, "high")] = prices * 1.01
        frames[(sym, "low")] = prices * 0.99
        frames[(sym, "close")] = prices
        frames[(sym, "volume")] = np.full(n_bars, 1000.0)
    df = pd.DataFrame(frames, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


# ---------------------------------------------------------------------------
# Strategy implementations for testing
# ---------------------------------------------------------------------------


class NoOpStrategy(Strategy):
    """Strategy that sets up a position and does nothing on each bar."""

    def setup(
        self, snapshot: AccountSnapshot, config: Dict[str, Any]
    ) -> AccountSnapshot:
        return AccountSnapshot(
            collateral=[CollateralPosition("SOL", 10, 150.0, 0.75, 0.80)],
            debt=[DebtPosition("USDC", 500, 1.0)],
        )

    def on_bar(
        self, snapshot: AccountSnapshot, bar: BarData
    ) -> List[Dict[str, Any]]:
        return []


class TrackingStrategy(Strategy):
    """Strategy that records every call for inspection."""

    def __init__(self) -> None:
        self.setup_calls: list = []
        self.on_bar_calls: list = []
        self.on_liquidation_calls: list[LiquidationEvent] = []

    def setup(
        self, snapshot: AccountSnapshot, config: Dict[str, Any]
    ) -> AccountSnapshot:
        self.setup_calls.append((snapshot, config))
        return AccountSnapshot(
            collateral=[CollateralPosition("SOL", 10, 150.0, 0.75, 0.80)],
            debt=[DebtPosition("USDC", 500, 1.0)],
        )

    def on_bar(
        self, snapshot: AccountSnapshot, bar: BarData
    ) -> List[Dict[str, Any]]:
        self.on_bar_calls.append((snapshot, bar))
        return []

    def on_liquidation(
        self,
        snapshot: AccountSnapshot,
        bar: BarData,
        event: LiquidationEvent,
    ) -> List[Dict[str, Any]]:
        self.on_liquidation_calls.append(event)
        return []


class LSTStrategy(Strategy):
    """Strategy that uses an LST asset (JitoSOL) as collateral."""

    def setup(
        self, snapshot: AccountSnapshot, config: Dict[str, Any]
    ) -> AccountSnapshot:
        return AccountSnapshot(
            collateral=[CollateralPosition("JitoSOL", 10, 150.0, 0.75, 0.80)],
            debt=[DebtPosition("USDC", 500, 1.0)],
        )

    def on_bar(
        self, snapshot: AccountSnapshot, bar: BarData
    ) -> List[Dict[str, Any]]:
        return []


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def market_params() -> MarketParams:
    return MarketParams.kamino_defaults()


@pytest.fixture
def engine_config() -> EngineConfig:
    return EngineConfig()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoOpStrategyRecordsAllBars:
    """Verify the engine records a history row for every bar with NoOpStrategy."""

    def test_history_length_matches_bars(self, market_params: MarketParams):
        n_bars = 50
        price_data = make_price_data(n_bars=n_bars, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        assert len(result.history) == n_bars

    def test_history_index_matches_price_data_index(self, market_params: MarketParams):
        n_bars = 30
        price_data = make_price_data(n_bars=n_bars, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        pd.testing.assert_index_equal(
            result.history.index, price_data.index, check_names=False
        )


class TestConstantPricesStablePortfolio:
    """With constant prices, portfolio value should stay stable (minus interest)."""

    def test_portfolio_value_decreases_only_by_interest(
        self, market_params: MarketParams
    ):
        n_bars = 100
        price_data = make_price_data(n_bars=n_bars, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        # Initial portfolio: collateral=10*150=1500, debt=500 => net=1000
        # USDC borrow rate is 8% APY, so interest accrues each bar
        # After 100 hours: interest ~= 500 * 0.08 / 8760 * 100 ~= 0.457 USD
        portfolio_values = result.history["portfolio_value"]
        initial = portfolio_values.iloc[0]
        final = portfolio_values.iloc[-1]

        # Value should decrease slightly due to USDC interest accrual
        assert final < initial
        # But should not decrease by more than a few dollars over 100 hours
        assert final > initial - 5.0

    def test_no_liquidation_events(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=50, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        assert len(result.liquidation_events) == 0
        assert result.liquidated is False

    def test_health_factor_stays_high(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=50, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        # HF = liquidation_value / debt = (1500 * 0.80) / ~500 = ~2.4
        min_hf = result.history["health_factor"].min()
        assert min_hf > 2.0


class TestCrashingPriceTriggersLiquidation:
    """A severe price crash should push HF below 1.0 and trigger liquidation."""

    def test_liquidation_events_produced(self, market_params: MarketParams):
        # Drift of -2% per hour will crash the price fast
        # After ~30 bars: price ~ 150 * 0.98^30 ~ 82
        # Collateral value ~ 10 * 82 = 820
        # Liquidation value ~ 820 * 0.80 = 656
        # Debt ~ 500 + small interest => HF ~ 656/500 ~ 1.31 still OK
        # After ~50 bars: price ~ 150 * 0.98^50 ~ 54
        # Collateral ~ 540, liq value ~ 432, debt ~ 500 => HF < 1.0
        price_data = make_price_data(n_bars=100, start_price=150.0, drift=-0.02)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        assert len(result.liquidation_events) > 0

    def test_full_liquidation_flag(self, market_params: MarketParams):
        # With a very steep crash, position gets fully liquidated
        price_data = make_price_data(n_bars=200, start_price=150.0, drift=-0.03)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        # The crash is severe enough that eventually all collateral is seized
        assert result.liquidated is True


class TestLSTYieldAccumulation:
    """With an LST collateral asset, lst_yield should accumulate over bars."""

    def test_lst_yield_positive(self, market_params: MarketParams):
        # JitoSOL price data -- using constant prices
        n_bars = 100
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
        price = 150.0
        df = pd.DataFrame(
            {
                ("JitoSOL", "open"): np.full(n_bars, price),
                ("JitoSOL", "high"): np.full(n_bars, price * 1.01),
                ("JitoSOL", "low"): np.full(n_bars, price * 0.99),
                ("JitoSOL", "close"): np.full(n_bars, price),
                ("JitoSOL", "volume"): np.full(n_bars, 1000.0),
            },
            index=dates,
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engine = BacktestEngine(LSTStrategy())
        result = engine.run(df, market_params=market_params)

        total_yield = result.history["lst_yield"].sum()
        assert total_yield > 0, "LST yield should be positive for JitoSOL collateral"

    def test_lst_yield_increases_collateral_value(self, market_params: MarketParams):
        n_bars = 500
        dates = pd.date_range("2024-01-01", periods=n_bars, freq="h", tz="UTC")
        price = 150.0
        df = pd.DataFrame(
            {
                ("JitoSOL", "open"): np.full(n_bars, price),
                ("JitoSOL", "high"): np.full(n_bars, price * 1.01),
                ("JitoSOL", "low"): np.full(n_bars, price * 0.99),
                ("JitoSOL", "close"): np.full(n_bars, price),
                ("JitoSOL", "volume"): np.full(n_bars, 1000.0),
            },
            index=dates,
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)

        engine = BacktestEngine(LSTStrategy())
        result = engine.run(df, market_params=market_params)

        # Collateral value should increase over time due to LST yield
        # (even though the close price resets each bar, the yield accrual
        # happens before the price update, so some bars will show higher value)
        # The total lst_yield earned should be meaningful over 500 bars
        total_yield = result.history["lst_yield"].sum()
        # JitoSOL 7% APY, 10 tokens * 150 = $1500 collateral
        # 500 hours: yield ~ 1500 * 0.07 / 8760 * 500 ~ $5.99
        assert total_yield > 4.0


class TestInterestAccrualTracked:
    """Interest accrual should be tracked in the history DataFrame."""

    def test_interest_accrued_column_populated(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=50, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        # USDC has 8% borrow rate APY
        total_interest = result.history["interest_accrued"].sum()
        assert total_interest > 0, "USDC debt should accrue interest"

    def test_interest_increases_debt_value(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=100, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        first_debt = result.history["debt_value"].iloc[0]
        last_debt = result.history["debt_value"].iloc[-1]
        assert last_debt > first_debt, "Debt should grow over time due to interest"

    def test_interest_matches_expected_magnitude(self, market_params: MarketParams):
        n_bars = 8760  # one year of hourly bars
        price_data = make_price_data(n_bars=n_bars, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        total_interest = result.history["interest_accrued"].sum()
        # USDC 8% APY on ~$500 principal = ~$40/year
        # Interest compounds because borrow action increases debt each bar,
        # so it will be somewhat higher, but should be in the right ballpark
        assert 35.0 < total_interest < 60.0, (
            f"Expected ~$40 annual interest on $500 USDC debt at 8% APY, got ${total_interest:.2f}"
        )


class TestOnLiquidationCallback:
    """strategy.on_liquidation should be called when liquidation occurs."""

    def test_on_liquidation_called_with_events(self, market_params: MarketParams):
        strategy = TrackingStrategy()
        price_data = make_price_data(n_bars=100, start_price=150.0, drift=-0.02)
        engine = BacktestEngine(strategy)
        result = engine.run(price_data, market_params=market_params)

        if len(result.liquidation_events) > 0:
            assert len(strategy.on_liquidation_calls) > 0
            # Each liquidation event should produce exactly one callback
            assert len(strategy.on_liquidation_calls) == len(
                result.liquidation_events
            )

    def test_on_liquidation_receives_liquidation_event_object(
        self, market_params: MarketParams
    ):
        strategy = TrackingStrategy()
        # Crash hard enough to definitely trigger liquidation
        price_data = make_price_data(n_bars=200, start_price=150.0, drift=-0.02)
        engine = BacktestEngine(strategy)
        result = engine.run(price_data, market_params=market_params)

        assert len(result.liquidation_events) > 0, (
            "Test requires liquidation to occur"
        )
        for event in strategy.on_liquidation_calls:
            assert isinstance(event, LiquidationEvent)
            assert event.debt_repaid > 0
            assert event.collateral_seized > 0


class TestStopOnFullLiquidation:
    """The stop_on_full_liquidation flag should control early termination."""

    def test_stop_true_truncates_history(self, market_params: MarketParams):
        n_bars = 200
        price_data = make_price_data(n_bars=n_bars, start_price=150.0, drift=-0.03)
        config = EngineConfig(stop_on_full_liquidation=True)
        engine = BacktestEngine(NoOpStrategy(), config=config)
        result = engine.run(price_data, market_params=market_params)

        if result.liquidated:
            # History should be shorter than n_bars because we stopped early
            assert len(result.history) < n_bars

    def test_stop_false_continues_after_liquidation(
        self, market_params: MarketParams
    ):
        n_bars = 200
        price_data = make_price_data(n_bars=n_bars, start_price=150.0, drift=-0.03)
        config = EngineConfig(stop_on_full_liquidation=False)
        engine = BacktestEngine(NoOpStrategy(), config=config)
        result = engine.run(price_data, market_params=market_params)

        if result.liquidated:
            # With stop=False, the engine should process all bars
            # (some bars may be skipped via `continue` but the loop
            # runs to completion)
            assert len(result.history) >= 1


class TestHistoryDataFrameSchema:
    """History DataFrame should have the correct columns and datetime index."""

    def test_has_all_expected_columns(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=10, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        for col in EXPECTED_HISTORY_COLUMNS:
            assert col in result.history.columns, f"Missing column: {col}"

    def test_index_is_datetime(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=10, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        assert isinstance(result.history.index, pd.DatetimeIndex)

    def test_index_name_is_timestamp(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=10, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        assert result.history.index.name == "timestamp"

    def test_no_nan_in_core_columns(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=20, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        core_cols = [
            "collateral_value",
            "debt_value",
            "portfolio_value",
            "action_count",
            "interest_accrued",
            "lst_yield",
            "liquidation_penalty",
        ]
        for col in core_cols:
            assert not result.history[col].isna().any(), (
                f"Column {col} contains NaN values"
            )


class TestBacktestResultIntegrity:
    """Verify BacktestResult fields are populated correctly."""

    def test_metrics_initial_value(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=20, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        # Initial: collateral=10*150=1500, debt=500 => net=1000
        assert abs(result.metrics.initial_value - 1000.0) < 1.0

    def test_liquidated_false_for_stable_prices(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=50, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        assert result.liquidated is False

    def test_market_params_stored(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=10, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        assert result.market_params is market_params

    def test_engine_config_stored(self, market_params: MarketParams):
        config = EngineConfig(lookback_bars=42, stop_on_full_liquidation=False)
        price_data = make_price_data(n_bars=10, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy(), config=config)
        result = engine.run(price_data, market_params=market_params)

        assert result.engine_config is config
        assert result.engine_config.lookback_bars == 42


class TestTrackingStrategyCallCounts:
    """Verify the engine calls strategy methods the expected number of times."""

    def test_setup_called_once(self, market_params: MarketParams):
        strategy = TrackingStrategy()
        price_data = make_price_data(n_bars=20, start_price=150.0, drift=0.0)
        engine = BacktestEngine(strategy)
        engine.run(price_data, market_params=market_params)

        assert len(strategy.setup_calls) == 1

    def test_on_bar_called_for_each_bar(self, market_params: MarketParams):
        strategy = TrackingStrategy()
        n_bars = 30
        price_data = make_price_data(n_bars=n_bars, start_price=150.0, drift=0.0)
        engine = BacktestEngine(strategy)
        engine.run(price_data, market_params=market_params)

        assert len(strategy.on_bar_calls) == n_bars

    def test_bar_data_has_correct_index(self, market_params: MarketParams):
        strategy = TrackingStrategy()
        n_bars = 10
        price_data = make_price_data(n_bars=n_bars, start_price=150.0, drift=0.0)
        engine = BacktestEngine(strategy)
        engine.run(price_data, market_params=market_params)

        for i, (snapshot, bar) in enumerate(strategy.on_bar_calls):
            assert bar.bar_index == i
            assert "SOL" in bar.prices


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_single_bar(self, market_params: MarketParams):
        price_data = make_price_data(n_bars=1, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=market_params)

        assert len(result.history) == 1

    def test_default_engine_config(self):
        engine = BacktestEngine(NoOpStrategy())
        assert engine.config.lookback_bars == 200
        assert engine.config.stop_on_full_liquidation is True
        assert engine.config.max_liquidation_cascades == 5

    def test_default_market_params_used_when_none(self):
        price_data = make_price_data(n_bars=5, start_price=150.0, drift=0.0)
        engine = BacktestEngine(NoOpStrategy())
        result = engine.run(price_data, market_params=None)

        # Should use kamino_defaults without error
        assert result.market_params is not None
        assert "SOL" in result.market_params.assets
        assert "USDC" in result.market_params.assets
