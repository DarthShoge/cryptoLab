"""Kamino Lending Strategy Backtester - Streamlit UI."""

import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from arblab.backtest.app_helpers import (
    DEFAULT_STRATEGY,
    EXCHANGE_SYMBOLS,
    LEVERAGE_LOOP_STRATEGY,
    SOL_SUPERTREND_SHORT_STRATEGY,
    build_price_configs,
    build_sol_supertrend_short_config,
    position_value_chart_data,
    run_selected_backtest,
    run_selected_grid_search,
)
from arblab.backtest.data import fetch_ohlcv
from arblab.backtest.market import MarketParams

st.set_page_config(page_title="Kamino Backtester", layout="wide")
st.title("Kamino Lending Strategy Backtester")

MAX_CHART_POINTS = 200


def _downsample(df: pd.DataFrame, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame:
    """Reduce chart data to *max_points* rows using evenly-spaced sampling."""
    if len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points, dtype=int)
    return df.iloc[idx]


@st.cache_data(show_spinner="Fetching price data...")
def _fetch_cached(strategy_name: str, collateral_symbol: str, timeframe: str, start: str, end: str, exchange_id: str):
    symbols = build_price_configs(strategy_name, collateral_symbol)
    return fetch_ohlcv(symbols=symbols, timeframe=timeframe, start=start, end=end, exchange_id=exchange_id)


@st.cache_data(show_spinner="Running backtest...")
def _run_backtest(strategy_name, price_data, strategy_config, _market_params):
    return run_selected_backtest(strategy_name, price_data, strategy_config, _market_params)


@st.cache_data(show_spinner="Running grid search...")
def _run_grid_search(strategy_name, price_data, param_grid, base_config, _market_params, sort_metric):
    return run_selected_grid_search(
        strategy_name=strategy_name, price_data=price_data, param_grid=param_grid,
        base_config=base_config, market_params=_market_params, sort_metric=sort_metric,
    )

# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.header("Data Settings")
exchange_id = st.sidebar.selectbox("Exchange", ["binance", "bybit", "okx"], index=0)
timeframe = st.sidebar.selectbox("Base Timeframe", ["1h"], index=0)
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2024-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-31"))

st.sidebar.header("Strategy")
strategy_name = st.sidebar.selectbox(
    "Strategy",
    [LEVERAGE_LOOP_STRATEGY, SOL_SUPERTREND_SHORT_STRATEGY],
    index=[LEVERAGE_LOOP_STRATEGY, SOL_SUPERTREND_SHORT_STRATEGY].index(DEFAULT_STRATEGY),
)

if strategy_name == SOL_SUPERTREND_SHORT_STRATEGY:
    collateral_symbol = "SOL"
    debt_symbol = "USDC"
else:
    st.sidebar.header("Assets")
    collateral_symbol = st.sidebar.selectbox(
        "Collateral Asset",
        ["SOL", "JitoSOL", "mSOL", "ETH"],
        index=0,
    )
    debt_symbol = st.sidebar.selectbox(
        "Debt Asset",
        ["USDC", "USDT"],
        index=0,
    )

hedge_enabled = False

if strategy_name == LEVERAGE_LOOP_STRATEGY:
    st.sidebar.header("Leverage Loop Parameters")
    initial_collateral = st.sidebar.number_input(
        "Initial Collateral (tokens)", value=100.0, min_value=1.0, step=10.0
    )
    num_loops = st.sidebar.slider("Leverage Loops", 1, 6, 3)
    loop_utilization = st.sidebar.slider(
        "Loop Utilization", 0.5, 0.95, 0.85, step=0.05
    )
    target_hf = st.sidebar.slider("Target HF", 1.1, 2.0, 1.3, step=0.05)
    rebalance_hf_low = st.sidebar.slider(
        "Delever Trigger (HF low)", 1.0, 1.5, 1.15, step=0.05
    )
    rebalance_hf_high = st.sidebar.slider(
        "Re-lever Trigger (HF high)", 1.5, 5.0, 2.5, step=0.1
    )

    st.sidebar.header("Hedge / Take Profit")
    hedge_enabled = st.sidebar.checkbox("Enable Hedge", value=False)
    hedge_hf_trigger = 2.5
    hedge_fraction = 1.0
    if hedge_enabled:
        hedge_hf_trigger = st.sidebar.slider(
            "Hedge HF Trigger", 1.5, 5.0, 2.5, step=0.1
        )
        hedge_fraction = st.sidebar.slider(
            "Hedge Fraction", 0.1, 1.0, 1.0, step=0.1
        )
else:
    st.sidebar.header("SOL Supertrend Parameters")
    initial_collateral = st.sidebar.number_input(
        "Initial SOL Collateral", value=100.0, min_value=1.0, step=10.0
    )
    supertrend_atr_period = st.sidebar.number_input(
        "Supertrend ATR Period", value=10, min_value=2, max_value=100, step=1
    )
    supertrend_multiplier = st.sidebar.slider(
        "Supertrend Multiplier", 1.0, 8.0, 3.0, step=0.25
    )
    target_bullish_hf = st.sidebar.slider("Target Bullish HF", 1.1, 2.0, 1.35, step=0.05)
    min_rebalance_hf = st.sidebar.slider("Minimum Rebalance HF", 1.05, 2.0, 1.25, step=0.05)
    max_usdc_debt_to_equity = st.sidebar.slider("Max USDC Debt / Equity", 0.25, 2.0, 1.0, step=0.25)
    rebalance_threshold = st.sidebar.slider("Rebalance Threshold", 0.01, 0.25, 0.10, step=0.01)
    rebalance_cooldown_bars = st.sidebar.slider("Cooldown Bars", 0, 24, 4)
    full_short_lower_bound = st.sidebar.slider(
        "Full Short Lower Bound", 0.75, 1.5, 1.0, step=0.05
    )
    full_short_upper_bound = st.sidebar.slider(
        "Full Short Upper Bound", 1.0, 2.5, 1.5, step=0.05
    )

st.sidebar.header("Market Overrides")
borrow_rate = st.sidebar.slider(
    "USDC Borrow Rate APY", 0.0, 0.30, 0.08, step=0.01, format="%.2f"
)
swap_fee_bps = st.sidebar.slider(
    "Swap Fee (bps)", 0.0, 100.0, 10.0, step=1.0
)
lst_apy = 0.07
if strategy_name == LEVERAGE_LOOP_STRATEGY:
    lst_apy = st.sidebar.slider(
        "LST Staking APY", 0.0, 0.15, 0.07, step=0.01, format="%.2f"
    )

st.sidebar.header("Mode")
mode = st.sidebar.radio("Run Mode", ["Single Backtest", "Grid Optimization"])

run_btn = st.sidebar.button("Run", type="primary")

# ── Optimization params ──────────────────────────────────────────────────

if mode == "Grid Optimization":
    st.sidebar.subheader("Grid Search Ranges")
    if strategy_name == LEVERAGE_LOOP_STRATEGY:
        loop_range = st.sidebar.text_input("num_loops (comma-sep)", "2,3,4")
        target_hf_range = st.sidebar.text_input(
            "target_hf (comma-sep)", "1.2,1.3,1.5"
        )
        if hedge_enabled:
            hedge_hf_range = st.sidebar.text_input(
                "hedge_hf_trigger (comma-sep)", "2.0,2.5,3.0"
            )
            hedge_frac_range = st.sidebar.text_input(
                "hedge_fraction (comma-sep)", "0.5,0.75,1.0"
            )
    else:
        atr_period_range = st.sidebar.text_input("supertrend_atr_period", "7,10,14")
        multiplier_range = st.sidebar.text_input("supertrend_multiplier", "2.0,3.0,4.0")
        threshold_range = st.sidebar.text_input("rebalance_threshold", "0.05,0.10,0.15")

# ── Main ─────────────────────────────────────────────────────────────────

if run_btn:
    # Build market params with overrides
    mp = MarketParams.kamino_defaults()
    assets = dict(mp.assets)
    from dataclasses import replace as dreplace

    if debt_symbol in assets:
        assets[debt_symbol] = dreplace(
            assets[debt_symbol], borrow_rate_apy=borrow_rate
        )
    for sym in ["JitoSOL", "mSOL"]:
        if sym in assets:
            assets[sym] = dreplace(assets[sym], lst_base_apy=lst_apy)
    mp = dreplace(mp, assets=assets, swap_fee_bps=swap_fee_bps)

    # Fetch price data (cached)
    try:
        price_data = _fetch_cached(
            strategy_name=strategy_name,
            collateral_symbol=collateral_symbol,
            timeframe=timeframe,
            start=str(start_date),
            end=str(end_date),
            exchange_id=exchange_id,
        )
    except Exception as e:
        st.error(f"Failed to fetch price data: {e}")
        st.stop()

    if price_data.empty:
        st.warning("No price data returned. Check date range and exchange.")
        st.stop()

    st.info(f"Loaded {len(price_data)} bars from {price_data.index[0]} to {price_data.index[-1]}")

    # Get initial price from first bar
    initial_price = float(price_data.iloc[0][(collateral_symbol, "close")])

    # Buy-and-hold benchmark: value of simply holding the initial collateral
    buy_hold = initial_collateral * price_data[(collateral_symbol, "close")]
    buy_hold.name = "Buy & Hold"

    # Extract asset params from market config
    coll_cfg = mp.assets.get(collateral_symbol)
    debt_cfg = mp.assets.get(debt_symbol)

    if strategy_name == LEVERAGE_LOOP_STRATEGY:
        strategy_config = {
            "collateral_symbol": collateral_symbol,
            "debt_symbol": debt_symbol,
            "initial_collateral": initial_collateral,
            "num_loops": num_loops,
            "loop_utilization": loop_utilization,
            "target_hf": target_hf,
            "rebalance_hf_low": rebalance_hf_low,
            "rebalance_hf_high": rebalance_hf_high,
            "initial_collateral_price": initial_price,
            "initial_debt_price": 1.0,
            "hedge_enabled": hedge_enabled,
            "hedge_hf_trigger": hedge_hf_trigger,
            "hedge_fraction": hedge_fraction,
            # Asset params from market config
            "ltv": coll_cfg.ltv if coll_cfg else 0.75,
            "liquidation_threshold": coll_cfg.liquidation_threshold if coll_cfg else 0.80,
            "borrow_factor": debt_cfg.borrow_factor if debt_cfg else 1.0,
        }
    else:
        strategy_config = build_sol_supertrend_short_config(
            price_data=price_data,
            initial_sol_collateral=initial_collateral,
            supertrend_atr_period=supertrend_atr_period,
            supertrend_multiplier=supertrend_multiplier,
            target_bullish_hf=target_bullish_hf,
            min_rebalance_hf=min_rebalance_hf,
            max_usdc_debt_to_equity=max_usdc_debt_to_equity,
            rebalance_threshold=rebalance_threshold,
            rebalance_cooldown_bars=rebalance_cooldown_bars,
            swap_fee_bps=mp.swap_fee_bps,
            full_short_lower_bound=full_short_lower_bound,
            full_short_upper_bound=full_short_upper_bound,
        )

    if mode == "Single Backtest":
        result = _run_backtest(strategy_name, price_data, strategy_config, mp)

        # Summary metrics
        m = result.metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{m.total_return_pct:+.2f}%")
        col2.metric("Max Drawdown", f"{m.max_drawdown_pct:.2f}%")
        col3.metric("Sharpe Ratio", f"{m.sharpe_ratio:.3f}")
        col4.metric("Sortino Ratio", f"{m.sortino_ratio:.3f}")

        # Risk Metrics Row
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Min HF", f"{m.min_health_factor:.3f}")
        col6.metric("Liquidations", str(m.total_liquidations))
        if not result.history.empty:
            final = result.history.iloc[-1]
            col7.metric("Borrow LTV (Kamino)", f"{final['borrow_ltv'] * 100:.2f}%")
            col8.metric("Liquidation LTV", f"{final['liquidation_ltv'] * 100:.2f}%")

        # Cost/Revenue Row
        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Interest Paid", f"${m.total_interest_paid:,.2f}")
        col10.metric("LST Yield", f"${m.total_lst_yield_earned:,.2f}")
        if not result.history.empty:
            col11.metric("Current LTV", f"{final['current_ltv'] * 100:.2f}%")
        # col12 empty for now

        # Hedge Metrics Row (if enabled)
        if hedge_enabled:
            col13, col14 = st.columns(2)
            col13.metric("Total Cash Hedged", f"${m.total_cash_hedged:,.2f}")
            col14.metric("Max Cash Reserve", f"${m.max_cash_reserve:,.2f}")

        if result.liquidated:
            st.error("Position was fully liquidated!")

        # Charts
        if not result.history.empty:
            st.subheader("Portfolio Value")
            pv_data = {
                "Strategy": result.history["portfolio_value"],
                "Buy & Hold": buy_hold.reindex(result.history.index, method="ffill"),
            }
            if hedge_enabled and "cash_reserve" in result.history:
                pv_data["Cash Reserve"] = result.history["cash_reserve"]
            pv_df = pd.DataFrame(pv_data)
            st.line_chart(_downsample(pv_df))

            st.subheader("Position Values")
            pos_df = position_value_chart_data(result.history)
            if not pos_df.empty:
                st.line_chart(_downsample(pos_df))
                with st.expander("Position Value Reconciliation"):
                    reconciliation = result.history[
                        ["collateral_value", "debt_value", "portfolio_value"]
                    ].copy()
                    reconciliation["net_collateral_minus_debt"] = (
                        reconciliation["collateral_value"]
                        - reconciliation["debt_value"]
                    )
                    st.dataframe(_downsample(reconciliation))

            st.subheader("Health Factor")
            hf_df = result.history[["health_factor"]].copy()
            hf_df["HF=1.0"] = 1.0
            hf_df["HF=1.5"] = 1.5
            # Cap display for readability
            hf_df["health_factor"] = hf_df["health_factor"].clip(upper=5.0)
            st.line_chart(_downsample(hf_df))

            st.subheader("Asset Prices")
            close_cols = [
                c for c in price_data.columns if c[1] == "close"
            ]
            price_display = price_data[close_cols].droplevel(1, axis=1)
            st.line_chart(_downsample(price_display))

            # Expandable details
            with st.expander("Interest & Yield Breakdown"):
                cum_interest = result.history["interest_accrued"].cumsum()
                cum_yield = result.history["lst_yield"].cumsum()
                breakdown = pd.DataFrame({
                    "Cumulative Interest Paid": cum_interest,
                    "Cumulative LST Yield": cum_yield,
                    "Net (Yield - Interest)": cum_yield - cum_interest,
                })
                st.line_chart(_downsample(breakdown))

            if result.liquidation_events:
                with st.expander("Liquidation Events"):
                    liq_rows = []
                    for e in result.liquidation_events:
                        liq_rows.append({
                            "Time": str(e.timestamp),
                            "Debt Repaid": f"{e.debt_repaid:.4f} {e.debt_symbol}",
                            "Debt Repaid USD": f"${e.debt_repaid_usd:,.2f}",
                            "Collateral Seized": f"{e.collateral_seized:.4f} {e.collateral_symbol}",
                            "Collateral Seized USD": f"${e.collateral_seized_usd:,.2f}",
                            "Bonus": f"{e.bonus_pct * 100:.1f}%",
                            "Resulting HF": f"{e.resulting_hf:.3f}",
                        })
                    st.dataframe(pd.DataFrame(liq_rows))

            if result.strategy_events:
                with st.expander("Strategy Events"):
                    st.dataframe(pd.DataFrame(result.strategy_events))

            with st.expander("Full History"):
                st.dataframe(result.history)

    else:  # Grid Optimization
        try:
            if strategy_name == LEVERAGE_LOOP_STRATEGY:
                loops = [int(x.strip()) for x in loop_range.split(",")]
                hfs = [float(x.strip()) for x in target_hf_range.split(",")]
                param_grid = {"num_loops": loops, "target_hf": hfs}
                if hedge_enabled:
                    hedge_hfs = [float(x.strip()) for x in hedge_hf_range.split(",")]
                    hedge_fracs = [float(x.strip()) for x in hedge_frac_range.split(",")]
                    param_grid["hedge_hf_trigger"] = hedge_hfs
                    param_grid["hedge_fraction"] = hedge_fracs
            else:
                param_grid = {
                    "supertrend_atr_period": [int(x.strip()) for x in atr_period_range.split(",")],
                    "supertrend_multiplier": [float(x.strip()) for x in multiplier_range.split(",")],
                    "rebalance_threshold": [float(x.strip()) for x in threshold_range.split(",")],
                }
        except ValueError:
            st.error("Invalid grid search ranges. Use comma-separated numbers.")
            st.stop()

        total_combos = 1
        for v in param_grid.values():
            total_combos *= len(v)
        st.info(f"Running {total_combos} parameter combinations...")

        opt_result = _run_grid_search(
            strategy_name=strategy_name,
            price_data=price_data,
            param_grid=param_grid,
            base_config=strategy_config,
            _market_params=mp,
            sort_metric="sortino_ratio",
        )

        st.subheader("Optimization Results")
        highlight_max_cols = ["sortino_ratio", "total_return_pct"]
        if "tracking_gap_vs_sol_pct" in opt_result.comparison_df:
            highlight_max_cols.append("tracking_gap_vs_sol_pct")
        st.dataframe(
            opt_result.comparison_df.style.highlight_max(
                subset=highlight_max_cols, color="lightgreen"
            ).highlight_min(subset=["max_drawdown_pct"], color="lightgreen"),
            use_container_width=True,
        )

        # Show best result details
        best = opt_result.best_result
        best_label = (
            "Best Result (by SOL benchmark tier, then Sortino)"
            if strategy_name == SOL_SUPERTREND_SHORT_STRATEGY
            else "Best Result (by Sortino)"
        )
        st.subheader(best_label)
        bm = best.metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{bm.total_return_pct:+.2f}%")
        col2.metric("Max Drawdown", f"{bm.max_drawdown_pct:.2f}%")
        col3.metric("Sharpe Ratio", f"{bm.sharpe_ratio:.3f}")
        col4.metric("Sortino Ratio", f"{bm.sortino_ratio:.3f}")

        # Risk Metrics Row
        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Min HF", f"{bm.min_health_factor:.3f}")
        col6.metric("Liquidations", str(bm.total_liquidations))
        if not best.history.empty:
            final = best.history.iloc[-1]
            col7.metric("Borrow LTV (Kamino)", f"{final['borrow_ltv'] * 100:.2f}%")
            col8.metric("Liquidation LTV", f"{final['liquidation_ltv'] * 100:.2f}%")

        # Cost/Revenue Row
        col9, col10, col11, col12 = st.columns(4)
        col9.metric("Interest Paid", f"${bm.total_interest_paid:,.2f}")
        col10.metric("LST Yield", f"${bm.total_lst_yield_earned:,.2f}")
        if not best.history.empty:
            col11.metric("Current LTV", f"{final['current_ltv'] * 100:.2f}%")
        # col12 empty for now

        # Hedge Metrics Row (if enabled)
        if hedge_enabled:
            col13, col14 = st.columns(2)
            col13.metric("Total Cash Hedged", f"${bm.total_cash_hedged:,.2f}")
            col14.metric("Max Cash Reserve", f"${bm.max_cash_reserve:,.2f}")

        if best.liquidated:
            st.error("Position was fully liquidated!")

        if not best.history.empty:
            st.subheader("Portfolio Value")
            pv_data = {
                "Strategy": best.history["portfolio_value"],
                "Buy & Hold": buy_hold.reindex(best.history.index, method="ffill"),
            }
            if hedge_enabled and "cash_reserve" in best.history:
                pv_data["Cash Reserve"] = best.history["cash_reserve"]
            pv_df = pd.DataFrame(pv_data)
            st.line_chart(_downsample(pv_df))

            st.subheader("Position Values")
            pos_df = position_value_chart_data(best.history)
            if not pos_df.empty:
                st.line_chart(_downsample(pos_df))
                with st.expander("Position Value Reconciliation"):
                    reconciliation = best.history[
                        ["collateral_value", "debt_value", "portfolio_value"]
                    ].copy()
                    reconciliation["net_collateral_minus_debt"] = (
                        reconciliation["collateral_value"]
                        - reconciliation["debt_value"]
                    )
                    st.dataframe(_downsample(reconciliation))

            st.subheader("Health Factor")
            hf_df = best.history[["health_factor"]].copy()
            hf_df["HF=1.0"] = 1.0
            hf_df["HF=1.5"] = 1.5
            hf_df["health_factor"] = hf_df["health_factor"].clip(upper=5.0)
            st.line_chart(_downsample(hf_df))

            st.subheader("Asset Prices")
            close_cols = [
                c for c in price_data.columns if c[1] == "close"
            ]
            price_display = price_data[close_cols].droplevel(1, axis=1)
            st.line_chart(_downsample(price_display))

            with st.expander("Interest & Yield Breakdown"):
                cum_interest = best.history["interest_accrued"].cumsum()
                cum_yield = best.history["lst_yield"].cumsum()
                breakdown = pd.DataFrame({
                    "Cumulative Interest Paid": cum_interest,
                    "Cumulative LST Yield": cum_yield,
                    "Net (Yield - Interest)": cum_yield - cum_interest,
                })
                st.line_chart(_downsample(breakdown))

            if best.liquidation_events:
                with st.expander("Liquidation Events"):
                    liq_rows = []
                    for e in best.liquidation_events:
                        liq_rows.append({
                            "Time": str(e.timestamp),
                            "Debt Repaid": f"{e.debt_repaid:.4f} {e.debt_symbol}",
                            "Debt Repaid USD": f"${e.debt_repaid_usd:,.2f}",
                            "Collateral Seized": f"{e.collateral_seized:.4f} {e.collateral_symbol}",
                            "Collateral Seized USD": f"${e.collateral_seized_usd:,.2f}",
                            "Bonus": f"{e.bonus_pct * 100:.1f}%",
                            "Resulting HF": f"{e.resulting_hf:.3f}",
                        })
                    st.dataframe(pd.DataFrame(liq_rows))

            if best.strategy_events:
                with st.expander("Strategy Events"):
                    st.dataframe(pd.DataFrame(best.strategy_events))

            with st.expander("Full History"):
                st.dataframe(best.history)
