"""Kamino Lending Strategy Backtester - Streamlit UI."""

import os

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from arblab.backtest.data import OHLCVConfig, fetch_ohlcv
from arblab.backtest.engine import BacktestEngine, EngineConfig
from arblab.backtest.market import MarketParams
from arblab.backtest.optimizer import grid_search
from arblab.strategies.leverage_loop import LeverageLoopStrategy

st.set_page_config(page_title="Kamino Backtester", layout="wide")
st.title("Kamino Lending Strategy Backtester")

# ── Sidebar ──────────────────────────────────────────────────────────────

st.sidebar.header("Data Settings")
exchange_id = st.sidebar.selectbox("Exchange", ["binance", "bybit", "okx"], index=0)
timeframe = st.sidebar.selectbox("Timeframe", ["1h", "4h", "1d"], index=0)
start_date = st.sidebar.date_input("Start Date", value=pd.Timestamp("2024-01-01"))
end_date = st.sidebar.date_input("End Date", value=pd.Timestamp("2024-12-31"))

st.sidebar.header("Assets")
collateral_symbol = st.sidebar.selectbox(
    "Collateral Asset", ["SOL", "JitoSOL", "mSOL"], index=0
)
EXCHANGE_SYMBOLS = {
    "SOL": "SOL/USDT",
    "JitoSOL": "JITOSOL/USDT",
    "mSOL": "MSOL/USDT",
}
debt_symbol = st.sidebar.selectbox("Debt Asset", ["USDC", "USDT"], index=0)

st.sidebar.header("Strategy Parameters")
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

st.sidebar.header("Market Overrides")
borrow_rate = st.sidebar.slider(
    "USDC Borrow Rate APY", 0.0, 0.30, 0.08, step=0.01, format="%.2f"
)
lst_apy = st.sidebar.slider(
    "LST Staking APY", 0.0, 0.15, 0.07, step=0.01, format="%.2f"
)

st.sidebar.header("Mode")
mode = st.sidebar.radio("Run Mode", ["Single Backtest", "Grid Optimization"])

run_btn = st.sidebar.button("Run", type="primary")

# ── Optimization params ──────────────────────────────────────────────────

if mode == "Grid Optimization":
    st.sidebar.subheader("Grid Search Ranges")
    loop_range = st.sidebar.text_input("num_loops (comma-sep)", "2,3,4")
    target_hf_range = st.sidebar.text_input(
        "target_hf (comma-sep)", "1.2,1.3,1.5"
    )

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
    mp = dreplace(mp, assets=assets)

    # Fetch price data
    with st.spinner("Fetching price data..."):
        try:
            symbols = [
                OHLCVConfig(
                    symbol=EXCHANGE_SYMBOLS.get(collateral_symbol, f"{collateral_symbol}/USDT"),
                    display_name=collateral_symbol,
                )
            ]
            price_data = fetch_ohlcv(
                symbols=symbols,
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

    strategy = LeverageLoopStrategy()
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
    }

    if mode == "Single Backtest":
        with st.spinner("Running backtest..."):
            engine = BacktestEngine(strategy)
            result = engine.run(price_data, strategy_config, mp)

        # Summary metrics
        m = result.metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Return", f"{m.total_return_pct:+.2f}%")
        col2.metric("Max Drawdown", f"{m.max_drawdown_pct:.2f}%")
        col3.metric("Sharpe Ratio", f"{m.sharpe_ratio:.3f}")
        col4.metric("Sortino Ratio", f"{m.sortino_ratio:.3f}")

        col5, col6, col7, col8 = st.columns(4)
        col5.metric("Min HF", f"{m.min_health_factor:.3f}")
        col6.metric("Liquidations", str(m.total_liquidations))
        col7.metric("Interest Paid", f"${m.total_interest_paid:,.2f}")
        col8.metric("LST Yield", f"${m.total_lst_yield_earned:,.2f}")

        if result.liquidated:
            st.error("Position was fully liquidated!")

        # Charts
        if not result.history.empty:
            st.subheader("Portfolio Value")
            st.line_chart(result.history["portfolio_value"])

            st.subheader("Health Factor")
            hf_df = result.history[["health_factor"]].copy()
            hf_df["HF=1.0"] = 1.0
            hf_df["HF=1.5"] = 1.5
            # Cap display for readability
            hf_df["health_factor"] = hf_df["health_factor"].clip(upper=5.0)
            st.line_chart(hf_df)

            st.subheader("Asset Prices")
            close_cols = [
                c for c in price_data.columns if c[1] == "close"
            ]
            price_display = price_data[close_cols].droplevel(1, axis=1)
            st.line_chart(price_display)

            # Expandable details
            with st.expander("Interest & Yield Breakdown"):
                cum_interest = result.history["interest_accrued"].cumsum()
                cum_yield = result.history["lst_yield"].cumsum()
                breakdown = pd.DataFrame({
                    "Cumulative Interest Paid": cum_interest,
                    "Cumulative LST Yield": cum_yield,
                    "Net (Yield - Interest)": cum_yield - cum_interest,
                })
                st.line_chart(breakdown)

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

            with st.expander("Full History"):
                st.dataframe(result.history)

    else:  # Grid Optimization
        try:
            loops = [int(x.strip()) for x in loop_range.split(",")]
            hfs = [float(x.strip()) for x in target_hf_range.split(",")]
        except ValueError:
            st.error("Invalid grid search ranges. Use comma-separated numbers.")
            st.stop()

        param_grid = {"num_loops": loops, "target_hf": hfs}
        total_combos = len(loops) * len(hfs)
        st.info(f"Running {total_combos} parameter combinations...")

        with st.spinner(f"Running grid search ({total_combos} backtests)..."):
            opt_result = grid_search(
                strategy=strategy,
                price_data=price_data,
                param_grid=param_grid,
                base_config=strategy_config,
                market_params=mp,
                sort_metric="sharpe_ratio",
            )

        st.subheader("Optimization Results")
        st.dataframe(
            opt_result.comparison_df.style.highlight_max(
                subset=["sharpe_ratio", "total_return_pct"], color="lightgreen"
            ).highlight_min(subset=["max_drawdown_pct"], color="lightgreen"),
            use_container_width=True,
        )

        # Show best result details
        best = opt_result.best_result
        st.subheader("Best Result (by Sharpe)")
        bm = best.metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Return", f"{bm.total_return_pct:+.2f}%")
        col2.metric("Sharpe", f"{bm.sharpe_ratio:.3f}")
        col3.metric("Max DD", f"{bm.max_drawdown_pct:.2f}%")
        col4.metric("Min HF", f"{bm.min_health_factor:.3f}")

        if not best.history.empty:
            st.line_chart(best.history["portfolio_value"])
