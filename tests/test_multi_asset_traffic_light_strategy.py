"""Tests for the curated-universe multi-asset traffic-light strategy."""

from __future__ import annotations

import pandas as pd

from arblab.backtest.market import MarketParams
from arblab.backtest.strategy import BarData, PriceBar
from arblab.kamino_risk import AccountSnapshot
from arblab.strategies.multi_asset_traffic_light import MultiAssetTrafficLightStrategy


def _bar(history: pd.DataFrame, index: int = -1) -> BarData:
    row = history.iloc[index]
    prices = {
        symbol: PriceBar(
            symbol=symbol,
            open=float(row[(symbol, "open")]),
            high=float(row[(symbol, "high")]),
            low=float(row[(symbol, "low")]),
            close=float(row[(symbol, "close")]),
            volume=float(row[(symbol, "volume")]),
        )
        for symbol in history.columns.get_level_values(0).unique()
    }
    return BarData(
        timestamp=history.index[index],
        prices=prices,
        history=history.iloc[: index + 1 if index >= 0 else None],
        bar_index=index if index >= 0 else len(history) - 1,
        market_params=MarketParams.kamino_defaults(),
    )


def _history(sol: list[float], eth: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(sol), freq="h", tz="UTC")
    data = {}
    for symbol, values in {
        "SOL": sol,
        "ETH": eth,
        "JitoSOL": sol,
        "mSOL": sol,
        "USDC": [1.0] * len(sol),
        "USDT": [1.0] * len(sol),
    }.items():
        series = pd.Series(values, dtype=float)
        data[(symbol, "open")] = series.tolist()
        data[(symbol, "high")] = (series * 1.01).tolist()
        data[(symbol, "low")] = (series * 0.99).tolist()
        data[(symbol, "close")] = series.tolist()
        data[(symbol, "volume")] = [1_000.0] * len(sol)
    df = pd.DataFrame(data, index=dates)
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def test_setup_creates_cash_collateral_and_zero_asset_slots():
    strategy = MultiAssetTrafficLightStrategy()

    snapshot = strategy.setup(
        AccountSnapshot(collateral=[], debt=[]),
        {
            "initial_usdc": 10_000.0,
            "directional_symbols": ["SOL", "ETH"],
            "initial_prices": {"SOL": 100.0, "ETH": 2_000.0},
        },
    )

    assert {position.symbol for position in snapshot.collateral} == {
        "USDC",
        "SOL",
        "ETH",
    }
    assert {position.symbol for position in snapshot.debt} == {"USDC", "SOL", "ETH"}
    assert next(p.amount for p in snapshot.collateral if p.symbol == "USDC") == 10_000.0
    assert next(p.amount for p in snapshot.collateral if p.symbol == "SOL") == 0.0
    assert next(p.amount for p in snapshot.debt if p.symbol == "USDC") == 0.0
    assert next(p.amount for p in snapshot.debt if p.symbol == "ETH") == 0.0


def test_setup_can_start_with_asset_collateral_for_sol_equivalent_benchmark():
    strategy = MultiAssetTrafficLightStrategy()

    snapshot = strategy.setup(
        AccountSnapshot(collateral=[], debt=[]),
        {
            "initial_collateral_symbol": "SOL",
            "initial_collateral_amount": 100.0,
            "directional_symbols": ["SOL", "ETH"],
            "initial_prices": {"SOL": 150.0, "ETH": 2_000.0},
        },
    )

    assert next(p.amount for p in snapshot.collateral if p.symbol == "SOL") == 100.0
    assert next(p.amount for p in snapshot.collateral if p.symbol == "USDC") == 0.0


def test_on_bar_can_borrow_usdc_to_target_leveraged_long_exposure():
    strategy = MultiAssetTrafficLightStrategy()
    history = _history(
        sol=[100.0 + i for i in range(96)],
        eth=[2_000.0 - i for i in range(96)],
    )
    snapshot = strategy.setup(
        AccountSnapshot(collateral=[], debt=[]),
        {
            "initial_collateral_symbol": "SOL",
            "initial_collateral_amount": 100.0,
            "directional_symbols": ["SOL", "ETH"],
            "initial_prices": {"SOL": 100.0, "ETH": 2_000.0},
            "atr_period": 3,
            "supertrend_multiplier": 1.5,
            "timeframes": ("1h", "4h"),
            "min_long_green": 2,
            "max_short_green": -1,
            "target_long_fraction": 1.50,
            "target_short_fraction": 0.00,
            "rebalance_threshold": 0.01,
        },
    )

    actions = strategy.on_bar(snapshot, _bar(history))

    assert any(action["type"] == "borrow" and action["symbol"] == "USDC" for action in actions)
    assert any(action["type"] == "deposit_collateral" and action["symbol"] == "SOL" for action in actions)


def test_on_bar_rotates_from_usdc_to_strong_long_and_weak_short():
    strategy = MultiAssetTrafficLightStrategy()
    history = _history(
        sol=[100.0 + i for i in range(96)],
        eth=[2_000.0 - i for i in range(96)],
    )
    snapshot = strategy.setup(
        AccountSnapshot(collateral=[], debt=[]),
        {
            "initial_usdc": 10_000.0,
            "directional_symbols": ["SOL", "ETH"],
            "initial_prices": {"SOL": 100.0, "ETH": 2_000.0},
            "atr_period": 3,
            "supertrend_multiplier": 1.5,
            "timeframes": ("1h", "4h"),
            "min_long_green": 2,
            "max_short_green": 0,
            "target_long_fraction": 0.50,
            "target_short_fraction": 0.20,
            "rebalance_threshold": 0.01,
        },
    )

    actions = strategy.on_bar(snapshot, _bar(history))

    assert {"type": "withdraw_collateral", "symbol": "USDC"} in [
        {key: action[key] for key in ("type", "symbol")} for action in actions
    ]
    assert any(action["type"] == "deposit_collateral" and action["symbol"] == "SOL" for action in actions)
    assert any(action["type"] == "borrow" and action["symbol"] == "ETH" for action in actions)
    assert strategy.history_fields()["selected_long"] == "SOL"
    assert strategy.history_fields()["selected_short"] == "ETH"


def test_on_bar_derisks_when_no_asset_has_enough_green_lights():
    strategy = MultiAssetTrafficLightStrategy()
    history = _history(
        sol=[100.0 + i for i in range(96)],
        eth=[2_000.0 - i for i in range(96)],
    )
    snapshot = strategy.setup(
        AccountSnapshot(collateral=[], debt=[]),
        {
            "initial_usdc": 10_000.0,
            "directional_symbols": ["SOL", "ETH"],
            "initial_prices": {"SOL": 100.0, "ETH": 2_000.0},
            "atr_period": 3,
            "supertrend_multiplier": 1.5,
            "timeframes": ("1h", "4h"),
            "min_long_green": 3,
            "max_short_green": -1,
            "target_long_fraction": 0.50,
            "target_short_fraction": 0.20,
            "rebalance_threshold": 0.01,
        },
    )
    snapshot = AccountSnapshot(
        collateral=[
            *[
                position
                for position in snapshot.collateral
                if position.symbol != "SOL"
            ],
            next(p for p in snapshot.collateral if p.symbol == "SOL").__class__(
                symbol="SOL",
                amount=25.0,
                price=150.0,
                ltv=0.75,
                liquidation_threshold=0.80,
            ),
        ],
        debt=snapshot.debt,
    )

    actions = strategy.on_bar(snapshot, _bar(history))

    assert any(action["type"] == "withdraw_collateral" and action["symbol"] == "SOL" for action in actions)
    assert any(action["type"] == "deposit_collateral" and action["symbol"] == "USDC" for action in actions)


def test_on_bar_can_use_precomputed_green_scores():
    strategy = MultiAssetTrafficLightStrategy()
    history = _history(
        sol=[100.0] * 8,
        eth=[2_000.0] * 8,
    )
    green_scores = pd.DataFrame(
        {"SOL": [2] * len(history), "ETH": [0] * len(history)},
        index=history.index,
    )
    snapshot = strategy.setup(
        AccountSnapshot(collateral=[], debt=[]),
        {
            "initial_usdc": 10_000.0,
            "directional_symbols": ["SOL", "ETH"],
            "initial_prices": {"SOL": 100.0, "ETH": 2_000.0},
            "green_scores": green_scores,
            "min_long_green": 2,
            "max_short_green": 0,
            "target_long_fraction": 0.50,
            "target_short_fraction": 0.20,
            "rebalance_threshold": 0.01,
        },
    )

    actions = strategy.on_bar(snapshot, _bar(history))

    assert any(action["type"] == "deposit_collateral" and action["symbol"] == "SOL" for action in actions)
    assert any(action["type"] == "borrow" and action["symbol"] == "ETH" for action in actions)


def test_drawdown_governor_caps_long_exposure_after_usd_equity_drawdown():
    strategy = MultiAssetTrafficLightStrategy()
    history = _history(
        sol=[100.0 + i for i in range(96)],
        eth=[2_000.0] * 96,
    )
    config = {
        "initial_collateral_symbol": "SOL",
        "initial_collateral_amount": 100.0,
        "directional_symbols": ["SOL", "ETH"],
        "initial_prices": {"SOL": 100.0, "ETH": 2_000.0},
        "atr_period": 3,
        "supertrend_multiplier": 1.5,
        "timeframes": ("1h", "4h"),
        "min_long_green": 2,
        "max_short_green": -1,
        "target_long_fraction": 1.50,
        "target_short_fraction": 0.00,
        "rebalance_threshold": 0.01,
        "enable_drawdown_governor": True,
        "drawdown_exposure_tiers": [
            {"drawdown": 0.0, "target_long_fraction": 1.50},
            {"drawdown": 0.20, "target_long_fraction": 1.00},
        ],
    }
    high_snapshot = strategy.setup(AccountSnapshot(collateral=[], debt=[]), config)
    strategy.on_bar(high_snapshot, _bar(history))

    low_snapshot = AccountSnapshot(
        collateral=[
            position.__class__(
                symbol=position.symbol,
                amount=position.amount,
                price=70.0 if position.symbol == "SOL" else position.price,
                ltv=position.ltv,
                liquidation_threshold=position.liquidation_threshold,
            )
            for position in high_snapshot.collateral
        ],
        debt=high_snapshot.debt,
    )

    strategy.on_bar(low_snapshot, _bar(history))

    fields = strategy.history_fields()
    assert fields["equity_drawdown_pct"] >= 30.0
    assert fields["target_long_fraction"] == 1.0


def test_signal_governor_caps_exposure_before_equity_drawdown():
    strategy = MultiAssetTrafficLightStrategy()
    history = _history(
        sol=[100.0] * 8,
        eth=[2_000.0] * 8,
    )
    green_scores = pd.DataFrame(
        {"SOL": [3] * len(history), "ETH": [1] * len(history)},
        index=history.index,
    )
    snapshot = strategy.setup(
        AccountSnapshot(collateral=[], debt=[]),
        {
            "initial_collateral_symbol": "SOL",
            "initial_collateral_amount": 100.0,
            "directional_symbols": ["SOL", "ETH"],
            "initial_prices": {"SOL": 100.0, "ETH": 2_000.0},
            "green_scores": green_scores,
            "min_long_green": 3,
            "max_short_green": -1,
            "target_long_fraction": 1.10,
            "target_short_fraction": 0.00,
            "rebalance_threshold": 0.01,
            "enable_signal_governor": True,
            "signal_exposure_tiers": [
                {"green": 4, "target_long_fraction": 1.10},
                {"green": 3, "target_long_fraction": 1.00},
            ],
        },
    )

    strategy.on_bar(snapshot, _bar(history))

    fields = strategy.history_fields()
    assert fields["long_green"] == 3
    assert fields["target_long_fraction"] == 1.0
    assert fields["equity_drawdown_pct"] == 0.0
