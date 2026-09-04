"""Reusable Streamlit explorer for strategy report artifacts."""

from __future__ import annotations

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from arblab.backtest.report_explorer import (
    build_buy_hold_frame,
    build_composition_frame,
    build_metric_frame,
    build_temperature_frame,
    build_timeline_frame,
    chart_debt_values_negative,
    default_report_index,
    default_strategy_selection,
    discover_report_dirs,
    final_composition_table,
    history_risk_stats,
    history_label_options,
    history_selection_for_summary,
    load_price_cache,
    load_report_bundle,
    portfolio_overview_table,
    regime_date_bounds,
    max_drawdown_series,
    slice_regime,
)


st.set_page_config(page_title="Strategy Report Explorer", layout="wide")
st.title("Strategy Report Explorer")

MAX_CHART_POINTS = 800
REPORT_ROOT = Path("reports")


def _downsample(df: pd.DataFrame, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    return df.iloc[pd.RangeIndex(0, len(df), max(1, len(df) // max_points)).to_numpy()]


@st.cache_data(show_spinner=False)
def _report_dirs() -> list[str]:
    return [str(path) for path in discover_report_dirs(REPORT_ROOT)]


@st.cache_data(show_spinner="Loading report artifacts...")
def _load_report(path: str):
    return load_report_bundle(Path(path))


@st.cache_data(show_spinner=False)
def _load_prices():
    return load_price_cache(Path("notebooks/.price_cache"), ["SOL", "ETH"])


def _format_money(value: object) -> str:
    return f"${float(value):,.2f}"


def _format_number(value: object, digits: int = 3) -> str:
    return f"{float(value):,.{digits}f}"


def _metric_delta(row: pd.Series, benchmark: pd.Series, column: str, digits: int = 3) -> str:
    if column not in row or column not in benchmark:
        return ""
    diff = float(row[column]) - float(benchmark[column])
    return f"{diff:+,.{digits}f}"


def _summary_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "name",
        "final_portfolio_value_usd",
        "final_sol_equiv",
        "max_drawdown_pct",
        "post_2024_drawdown_pct",
        "sortino_ratio",
        "min_health_factor",
        "bars_below_hf_1_5",
        "directional_overlap_count",
        "avg_target_long_fraction",
        "avg_target_short_fraction",
        "total_interest_paid",
    ]
    existing = [column for column in columns if column in summary.columns]
    return summary[existing].copy()


def _format_overview_table(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    money_columns = [
        "final_portfolio_value_usd",
        "total_interest_paid",
        "strategy_vs_buy_hold_usd",
        "strategy_vs_eth_buy_hold_usd",
        "sol_buy_hold_final_usd",
        "eth_buy_hold_final_usd",
    ]
    pct_columns = [
        "max_drawdown_pct",
        "post_2024_drawdown_pct",
        "buy_hold_max_drawdown_pct",
        "eth_buy_hold_max_drawdown_pct",
    ]
    number_columns = [
        "final_sol_equiv",
        "sortino_ratio",
        "sharpe_ratio_check",
        "information_ratio_vs_sol",
        "min_health_factor",
        "estimated_annualized_turnover_multiple",
    ]
    for column in money_columns:
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else _format_money(value))
    for column in pct_columns:
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{float(value):,.3f}%")
    for column in number_columns:
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else _format_number(value))
    for column in ["total_actions", "bars_below_hf_1_5", "total_liquidations", "directional_overlap_count"]:
        if column in formatted:
            formatted[column] = formatted[column].map(lambda value: "" if pd.isna(value) else f"{int(float(value)):,}")
    if "action_turnover_per_year" in formatted:
        formatted["action_turnover_per_year"] = formatted["action_turnover_per_year"].map(
            lambda value: "" if pd.isna(value) else f"{float(value):,.1f}"
        )
    if "estimated_annualized_turnover_multiple" in formatted:
        formatted["estimated_annualized_turnover_multiple"] = formatted[
            "estimated_annualized_turnover_multiple"
        ].map(lambda value: "" if pd.isna(value) else f"{float(value):,.2f}x")
    return formatted


def _annualized_volatility_pct(history: pd.DataFrame) -> float:
    if history.empty or "portfolio_value" not in history:
        return 0.0
    returns = history["portfolio_value"].astype(float).pct_change().dropna()
    if returns.empty:
        return 0.0
    return float(returns.std()) * (24.0 * 365.25) ** 0.5 * 100.0


def _value_or_fallback(value: object, fallback: float) -> object:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except TypeError:
        return value
    return value


def _core_stats_table(
    summary: pd.DataFrame,
    selected_names: list[str],
    selected_history_names: list[str],
    histories: dict[str, pd.DataFrame],
    prices: dict[str, pd.Series],
) -> pd.DataFrame:
    if summary.empty or "name" not in summary:
        return pd.DataFrame()
    history_by_summary = dict(zip(selected_names, selected_history_names))
    rows = []
    keyed = summary.assign(name=summary["name"].astype(str)).set_index("name", drop=False)
    for name in selected_names:
        if name not in keyed.index:
            continue
        row = keyed.loc[name]
        history = histories.get(history_by_summary.get(name, ""))
        fallback_stats = history_risk_stats(history, prices) if history is not None else {}
        rows.append(
            {
                "strategy": name,
                "final_sol": row.get("final_sol_equiv"),
                "max_dd_pct": row.get("max_drawdown_pct"),
                "sortino": row.get("sortino_ratio"),
                "sharpe": _value_or_fallback(
                    row.get("sharpe_ratio_check"),
                    fallback_stats.get("sharpe_ratio_check", 0.0),
                ),
                "ir_vs_sol": _value_or_fallback(
                    row.get("information_ratio_vs_sol"),
                    fallback_stats.get("information_ratio_vs_sol", 0.0),
                ),
                "volatility_pct": fallback_stats.get(
                    "annualized_volatility_pct",
                    _annualized_volatility_pct(history) if history is not None else 0.0,
                ),
                "turnover_x": row.get("estimated_annualized_turnover_multiple"),
                "actions": row.get("total_actions"),
                "actions_per_year": row.get("action_turnover_per_year"),
                "min_hf": row.get("min_health_factor"),
                "bars_hf_lt_1_5": row.get("bars_below_hf_1_5"),
                "liquidations": row.get("total_liquidations"),
                "overlap_bars": row.get("directional_overlap_count"),
                "interest_paid": row.get("total_interest_paid"),
            }
        )
    return pd.DataFrame(rows)


def _format_core_stats_table(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for column in ["final_sol", "sortino", "sharpe", "ir_vs_sol", "min_hf"]:
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):,.3f}"
            )
    for column in ["max_dd_pct", "volatility_pct"]:
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):,.2f}%"
            )
    if "turnover_x" in formatted:
        formatted["turnover_x"] = formatted["turnover_x"].map(
            lambda value: "" if pd.isna(value) else f"{float(value):,.2f}x"
        )
    if "actions_per_year" in formatted:
        formatted["actions_per_year"] = formatted["actions_per_year"].map(
            lambda value: "" if pd.isna(value) else f"{float(value):,.1f}"
        )
    for column in ["actions", "bars_hf_lt_1_5", "liquidations", "overlap_bars"]:
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{int(float(value)):,}"
            )
    if "interest_paid" in formatted:
        formatted["interest_paid"] = formatted["interest_paid"].map(
            lambda value: "" if pd.isna(value) else _format_money(value)
        )
    return formatted


def _history_regime_overview(history: pd.DataFrame) -> pd.DataFrame:
    regimes = {
        "full": (None, None),
        "2021_bull": ("2021-01-01", "2021-12-31 23:59:59+00:00"),
        "2022_crash": ("2022-01-01", "2022-12-31 23:59:59+00:00"),
        "2023_recovery": ("2023-01-01", "2023-12-31 23:59:59+00:00"),
        "2024_2026": ("2024-01-01", None),
        "2026_ytd": ("2026-01-01", None),
    }
    rows: list[dict[str, object]] = []
    for regime, (start, end) in regimes.items():
        window = slice_regime(history, start, end)
        if len(window) < 2 or "portfolio_value" not in window:
            continue
        start_value = float(window["portfolio_value"].iloc[0])
        end_value = float(window["portfolio_value"].iloc[-1])
        action_count = float(window.get("action_count", pd.Series(0.0, index=window.index)).sum())
        cooldown = window.get(
            "rebalance_cooldown_active",
            pd.Series(False, index=window.index),
        )
        rows.append(
            {
                "regime": regime,
                "start": window.index[0],
                "end": window.index[-1],
                "return_pct": (end_value / start_value - 1.0) * 100.0 if start_value > 0 else 0.0,
                "max_drawdown_pct": float(max_drawdown_series(window["portfolio_value"]).max()),
                "end_value_usd": end_value,
                "actions": int(action_count),
                "avg_target_long": float(window.get("target_long_fraction", pd.Series(0.0, index=window.index)).mean()),
                "avg_target_short": float(window.get("target_short_fraction", pd.Series(0.0, index=window.index)).mean()),
                "cooldown_share": float(cooldown.astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows)


def _history_traffic_state_overview(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty or "traffic_light_state" not in history:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for state, frame in history.groupby("traffic_light_state", dropna=False):
        cooldown = frame.get(
            "rebalance_cooldown_active",
            pd.Series(False, index=frame.index),
        )
        threshold = frame.get(
            "rebalance_threshold_pct",
            pd.Series(0.0, index=frame.index),
        )
        rows.append(
            {
                "state": state,
                "share": len(frame) / len(history),
                "avg_target_long": float(frame.get("target_long_fraction", pd.Series(0.0, index=frame.index)).mean()),
                "avg_target_short": float(frame.get("target_short_fraction", pd.Series(0.0, index=frame.index)).mean()),
                "avg_threshold_pct": float(threshold.mean()),
                "actions": int(frame.get("action_count", pd.Series(0.0, index=frame.index)).sum()),
                "cooldown_share": float(cooldown.astype(bool).mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("share", ascending=False).reset_index(drop=True)


def _format_regime_table(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for column in ["start", "end"]:
        if column in formatted:
            formatted[column] = pd.to_datetime(formatted[column]).dt.strftime("%Y-%m-%d")
    for column in ["return_pct", "max_drawdown_pct", "cooldown_share"]:
        if column in formatted:
            multiplier = 100.0 if column == "cooldown_share" else 1.0
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value) * multiplier:,.2f}%"
            )
    if "end_value_usd" in formatted:
        formatted["end_value_usd"] = formatted["end_value_usd"].map(
            lambda value: "" if pd.isna(value) else _format_money(value)
        )
    for column in ["avg_target_long", "avg_target_short"]:
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else _format_number(value)
            )
    return formatted


def _format_state_table(table: pd.DataFrame) -> pd.DataFrame:
    formatted = table.copy()
    for column in ["share", "cooldown_share"]:
        if column in formatted:
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value) * 100.0:,.2f}%"
            )
    for column in ["avg_target_long", "avg_target_short", "avg_threshold_pct"]:
        if column in formatted:
            suffix = "%" if column == "avg_threshold_pct" else ""
            formatted[column] = formatted[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):,.3f}{suffix}"
            )
    return formatted


def _render_portfolio_overview(
    bundle,
    summary: pd.DataFrame,
    selected_names: list[str],
    selected_history_names: list[str],
) -> None:
    if summary.empty:
        st.info("No summary data is available for this overview.")
        return

    histories = {
        name: bundle.histories[name]
        for name in selected_history_names
        if name in bundle.histories
    }
    prices = _load_prices()
    core_stats = _core_stats_table(summary, selected_names, selected_history_names, histories, prices)
    if not core_stats.empty:
        st.subheader("Core Risk / Turnover Stats")
        st.dataframe(
            _format_core_stats_table(core_stats),
            use_container_width=True,
            hide_index=True,
        )

    overview = portfolio_overview_table(summary, selected_names)
    if overview.empty:
        st.info("No selected strategy summary rows are available.")
    else:
        st.subheader("Portfolio Accounting")
        st.dataframe(_format_overview_table(overview), use_container_width=True, hide_index=True)

    benchmark_columns = [
        "name",
        "buy_hold_final_usd",
        "buy_hold_max_drawdown_pct",
        "strategy_vs_buy_hold_usd",
        "strategy_vs_buy_hold_sol",
        "eth_buy_hold_final_usd",
        "eth_buy_hold_max_drawdown_pct",
        "strategy_vs_eth_buy_hold_usd",
        "strategy_vs_eth_buy_hold_pct",
    ]
    existing_benchmark_columns = [column for column in benchmark_columns if column in summary.columns]
    if len(existing_benchmark_columns) > 1:
        benchmark = summary[summary["name"].astype(str).isin(selected_names)][existing_benchmark_columns]
        if not benchmark.empty:
            st.subheader("Benchmark Comparison")
            st.dataframe(_format_overview_table(benchmark), use_container_width=True, hide_index=True)

    if histories:
        st.subheader("Selected Strategy Final Composition")
        composition_tabs = st.tabs(list(histories))
        for tab, (name, history) in zip(composition_tabs, histories.items()):
            with tab:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.caption("Collateral")
                    st.dataframe(
                        final_composition_table(history, "collateral"),
                        use_container_width=True,
                        hide_index=True,
                    )
                with col_b:
                    st.caption("Debt")
                    st.dataframe(
                        final_composition_table(history, "debt"),
                        use_container_width=True,
                        hide_index=True,
                    )

        st.subheader("Selected Strategy Regime Stats")
        regime_tabs = st.tabs(list(histories))
        for tab, (name, history) in zip(regime_tabs, histories.items()):
            with tab:
                st.dataframe(
                    _format_regime_table(_history_regime_overview(history)),
                    use_container_width=True,
                    hide_index=True,
                )

        st.subheader("Selected Strategy Traffic-State Profile")
        state_tabs = st.tabs(list(histories))
        for tab, (name, history) in zip(state_tabs, histories.items()):
            with tab:
                state_table = _history_traffic_state_overview(history)
                if state_table.empty:
                    st.info("No traffic-light state history is available for this strategy.")
                else:
                    st.dataframe(
                        _format_state_table(state_table),
                        use_container_width=True,
                        hide_index=True,
                    )

    elif bundle.extra_tables:
        final_composition = bundle.extra_tables.get("final_composition")
        if final_composition is not None and not final_composition.empty:
            st.subheader("Report Final Composition")
            st.dataframe(final_composition, use_container_width=True, hide_index=True)

        traffic_states = bundle.extra_tables.get("traffic_states")
        if traffic_states is not None and not traffic_states.empty:
            st.subheader("Traffic-State Profile")
            st.dataframe(traffic_states, use_container_width=True, hide_index=True)

        if not bundle.regimes.empty:
            st.subheader("Regime Stats")
            st.dataframe(bundle.regimes, use_container_width=True, hide_index=True)


def _aligned_sol_price(index: pd.Index, prices: dict[str, pd.Series]) -> pd.Series:
    sol_price = prices.get("SOL")
    if sol_price is None or len(index) == 0:
        return pd.Series(dtype=float, index=index)
    return sol_price.reindex(index, method="ffill").astype(float)


def _composition_chart(
    composition: pd.DataFrame,
    prices: dict[str, pd.Series],
) -> alt.Chart:
    chart_frame = _downsample(composition).copy()
    chart_frame.index.name = "timestamp"
    composition_long = chart_frame.reset_index().melt(
        id_vars="timestamp",
        var_name="asset",
        value_name="value_usd",
    )
    area = (
        alt.Chart(composition_long)
        .mark_area(opacity=0.78)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y("value_usd:Q", title="Asset value (USD)", stack="zero"),
            color=alt.Color("asset:N", title="Asset"),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("asset:N", title="Asset"),
                alt.Tooltip("value_usd:Q", title="Value", format=",.2f"),
            ],
        )
    )

    sol_price = _aligned_sol_price(composition.index, prices).dropna()
    if sol_price.empty:
        return area.properties(height=320)

    price_frame = _downsample(
        pd.DataFrame({"sol_price": sol_price}, index=sol_price.index)
    ).copy()
    price_frame.index.name = "timestamp"
    price_data = price_frame.reset_index()
    line = (
        alt.Chart(price_data)
        .mark_line(color="#111827", strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y(
                "sol_price:Q",
                title="SOL price (USD)",
                axis=alt.Axis(orient="right"),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("sol_price:Q", title="SOL price", format=",.2f"),
            ],
        )
    )
    return alt.layer(area, line).resolve_scale(y="independent").properties(height=320)


def _temperature_chart(
    temperature: pd.DataFrame,
) -> alt.Chart:
    chart_frame = _downsample(temperature).copy()
    chart_frame.index.name = "timestamp"
    data = chart_frame.reset_index()

    temperature_bars = (
        alt.Chart(data)
        .mark_bar(opacity=0.44)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y(
                "net_temperature:Q",
                title="Net exposure temperature",
                scale=alt.Scale(domain=[-1.5, 1.5]),
            ),
            color=alt.Color(
                "net_temperature:Q",
                title="Temperature",
                scale=alt.Scale(
                    domain=[-1.0, 0.0, 1.5],
                    range=["#2563eb", "#e5e7eb", "#dc2626"],
                ),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip(
                    "net_temperature:Q",
                    title="Net temperature",
                    format=",.3f",
                ),
            ],
        )
    )
    sol_line = (
        alt.Chart(data)
        .mark_line(color="#111827", strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y(
                "sol_price:Q",
                title="SOL price (USD)",
                axis=alt.Axis(orient="right"),
            ),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("sol_price:Q", title="SOL price", format=",.2f"),
            ],
        )
    )
    return (
        alt.layer(temperature_bars, sol_line)
        .resolve_scale(y="independent")
        .properties(height=320)
    )


def _timeline_chart(timeline: pd.DataFrame, y_mode: str = "sol_price") -> alt.Chart:
    chart_frame = _downsample(timeline, max_points=1200).copy()
    primary_field = "strategy_pnl_usd" if y_mode == "strategy_pnl_usd" else "sol_price"
    primary_title = "Strategy PnL (USD)" if primary_field == "strategy_pnl_usd" else "SOL price (USD)"
    primary_columns = list(
        dict.fromkeys(["timestamp", primary_field, "strategy_pnl_usd", "sol_price"])
    )
    primary_data = chart_frame[
        [column for column in primary_columns if column in chart_frame.columns]
    ].drop_duplicates("timestamp")
    zoom = alt.selection_interval(bind="scales", encodings=["x"])
    primary_line = (
        alt.Chart(primary_data)
        .mark_line(color="#111827" if primary_field == "sol_price" else "#2563eb", strokeWidth=2)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y(f"{primary_field}:Q", title=primary_title),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("strategy_pnl_usd:Q", title="Strategy PnL", format=",.2f"),
                alt.Tooltip("sol_price:Q", title="SOL price", format=",.2f"),
            ],
        )
    )
    layers: list[alt.Chart] = [primary_line]
    if primary_field == "strategy_pnl_usd":
        sol_line = (
            alt.Chart(primary_data)
            .mark_line(color="#111827", strokeWidth=2, opacity=0.72)
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y(
                    "sol_price:Q",
                    title="SOL price (USD)",
                    axis=alt.Axis(orient="right"),
                ),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("strategy_pnl_usd:Q", title="Strategy PnL", format=",.2f"),
                    alt.Tooltip("sol_price:Q", title="SOL price", format=",.2f"),
                ],
            )
        )
        layers.append(sol_line)
    markers = (
        alt.Chart(chart_frame)
        .mark_point(filled=True, size=88, opacity=0.88)
        .encode(
            x=alt.X("timestamp:T", title=None),
            y=alt.Y(f"{primary_field}:Q", title=primary_title),
            color=alt.Color("event_family:N", title="Event"),
            shape=alt.Shape("event_family:N", title="Event"),
            tooltip=[
                alt.Tooltip("timestamp:T", title="Time"),
                alt.Tooltip("event_family:N", title="Family"),
                alt.Tooltip("event:N", title="Event"),
                alt.Tooltip("selected_long:N", title="Long"),
                alt.Tooltip("target_long_fraction:Q", title="Long target", format=",.3f"),
                alt.Tooltip("target_short_fraction:Q", title="Short target", format=",.3f"),
                alt.Tooltip("health_factor:Q", title="HF", format=",.3f"),
                alt.Tooltip("drawdown_pct:Q", title="Drawdown", format=",.2f"),
                alt.Tooltip("portfolio_value:Q", title="Portfolio value", format=",.2f"),
                alt.Tooltip("strategy_pnl_usd:Q", title="Strategy PnL", format=",.2f"),
                alt.Tooltip("sol_price:Q", title="SOL price", format=",.2f"),
                alt.Tooltip("snapshot_portfolio:N", title="Portfolio"),
                alt.Tooltip("snapshot_collateral_SOL:N", title="Collateral SOL"),
                alt.Tooltip("snapshot_collateral_ETH:N", title="Collateral ETH"),
                alt.Tooltip("snapshot_collateral_BTC:N", title="Collateral BTC"),
                alt.Tooltip("snapshot_collateral_USDC:N", title="Collateral USDC"),
                alt.Tooltip("snapshot_debt_SOL:N", title="Debt SOL"),
                alt.Tooltip("snapshot_debt_ETH:N", title="Debt ETH"),
                alt.Tooltip("snapshot_debt_BTC:N", title="Debt BTC"),
                alt.Tooltip("snapshot_debt_USDC:N", title="Debt USDC"),
            ],
        )
    )
    layers.append(markers)
    return (
        alt.layer(*layers)
        .resolve_scale(y="independent" if primary_field == "strategy_pnl_usd" else "shared")
        .add_params(zoom)
        .properties(height=360)
    )


def _render_kpis(summary: pd.DataFrame, names: list[str]) -> None:
    if summary.empty or not names:
        return
    benchmark = summary[summary["name"] == names[0]].iloc[0]
    cols = st.columns(len(names))
    for col, name in zip(cols, names):
        row = summary[summary["name"] == name].iloc[0]
        with col:
            st.subheader(str(name))
            st.metric(
                "Final USD",
                _format_money(row["final_portfolio_value_usd"]),
                _metric_delta(row, benchmark, "final_portfolio_value_usd", 2),
            )
            st.metric(
                "Final SOL",
                _format_number(row["final_sol_equiv"]),
                _metric_delta(row, benchmark, "final_sol_equiv"),
            )
            st.metric(
                "Max DD",
                f"{_format_number(row['max_drawdown_pct'])}%",
                _metric_delta(row, benchmark, "max_drawdown_pct"),
            )
            if "sortino_ratio" in row:
                st.metric(
                    "Sortino",
                    _format_number(row["sortino_ratio"]),
                    _metric_delta(row, benchmark, "sortino_ratio"),
                )


def _slice_histories(
    histories: dict[str, pd.DataFrame],
    names: list[str],
    start: str | None,
    end: str | None,
) -> dict[str, pd.DataFrame]:
    return {
        name: slice_regime(histories[name], start, end)
        for name in names
        if name in histories
    }


report_paths = _report_dirs()
if not report_paths:
    st.warning("No report directories with summary.csv or report.md were found.")
    st.stop()

if st.sidebar.button("Refresh reports"):
    _report_dirs.clear()
    _load_report.clear()
    st.rerun()

default_index = default_report_index(report_paths)

selected_path = st.sidebar.selectbox(
    "Report directory",
    report_paths,
    index=default_index,
    format_func=lambda value: Path(value).name,
)
bundle = _load_report(selected_path)

if bundle.summary.empty:
    st.warning("This report has no summary.csv. Markdown can still be viewed below.")
    if bundle.markdown:
        st.markdown(bundle.markdown)
    st.stop()

summary = bundle.summary
strategy_names = summary["name"].astype(str).tolist() if "name" in summary else []
history_options = history_label_options(summary, bundle.histories)

st.sidebar.header("Strategies")
default_selection = default_strategy_selection(strategy_names, history_options)
selected_names = st.sidebar.multiselect(
    "Compare",
    strategy_names,
    default=default_selection,
)
selected_history_names = history_selection_for_summary(selected_names, history_options)

st.sidebar.header("Regime")
regime_bounds = regime_date_bounds(bundle.regimes)
regime = st.sidebar.selectbox("Preset", list(regime_bounds), index=0)
start, end = regime_bounds[regime]
custom_range = st.sidebar.checkbox("Custom date range", value=False)
if custom_range and bundle.histories:
    first_history = next(iter(bundle.histories.values()))
    min_date = first_history.index.min().date()
    max_date = first_history.index.max().date()
    start_date, end_date = st.sidebar.date_input(
        "Dates",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    start = str(start_date)
    end = f"{end_date} 23:59:59+00:00"

st.caption(str(bundle.path))
_render_kpis(summary, selected_names)

tabs = st.tabs(["Overview", "Dynamics", "Composition", "Timeline", "Regimes", "Report", "Raw Tables"])

with tabs[0]:
    _render_portfolio_overview(bundle, summary, selected_names, selected_history_names)

with tabs[1]:
    if not bundle.histories or not selected_history_names:
        st.info("No local history CSVs are available for dynamic charts.")
    else:
        histories = _slice_histories(bundle.histories, selected_history_names, start, end)
        if not histories:
            st.info("No selected strategies have matching local history CSVs.")
            st.stop()
        prices = _load_prices()
        buy_hold = build_buy_hold_frame(histories, prices, ["SOL", "ETH"])
        if not buy_hold.empty:
            buy_hold = slice_regime(buy_hold, start, end)

            st.subheader("Portfolio Value vs Buy & Hold")
            portfolio_frame = build_metric_frame(histories, "portfolio_value")
            combined_portfolio = pd.concat([portfolio_frame, buy_hold], axis=1)
            st.line_chart(_downsample(combined_portfolio.dropna(how="all")))

            st.subheader("Normalized Value vs Buy & Hold")
            normalized_strategy = build_metric_frame(histories, "normalized_portfolio_value")
            normalized_buy_hold = buy_hold / buy_hold.iloc[0] * 100.0
            combined_normalized = pd.concat(
                [normalized_strategy, normalized_buy_hold],
                axis=1,
            )
            st.line_chart(_downsample(combined_normalized.dropna(how="all")))

        chart_cols = st.columns(2)
        chart_specs = [
            ("Normalized portfolio value", "normalized_portfolio_value"),
            ("Drawdown %", "drawdown_pct"),
            ("Health factor", "health_factor"),
            ("Target long fraction", "target_long_fraction"),
            ("Target short fraction", "target_short_fraction"),
            ("Realized vol %", "realized_vol_pct"),
            ("Equity drawdown signal %", "equity_drawdown_pct"),
            ("Recovery boost active", "recovery_boost_active"),
        ]
        for idx, (title, metric) in enumerate(chart_specs):
            frame = build_metric_frame(histories, metric)
            if frame.empty:
                continue
            with chart_cols[idx % 2]:
                st.subheader(title)
                st.line_chart(_downsample(frame))

        st.subheader("SOL Price Temperature")
        for name, history in histories.items():
            temperature = build_temperature_frame(history, prices)
            if temperature.empty:
                continue
            with st.expander(name, expanded=False):
                st.altair_chart(_temperature_chart(temperature), use_container_width=True)

        st.subheader("Position Values")
        for name, history in histories.items():
            with st.expander(name, expanded=False):
                position_columns = [
                    column
                    for column in history.columns
                    if column.endswith("_value")
                    and (
                        column.startswith("collateral_")
                        or column.startswith("debt_")
                        or column == "portfolio_value"
                    )
                ]
                if position_columns:
                    st.line_chart(
                        _downsample(chart_debt_values_negative(history[position_columns]))
                    )
                visible_columns = [
                    column
                    for column in [
                        "selected_long",
                        "long_green",
                        "target_long_fraction",
                        "target_short_fraction",
                        "equity_drawdown_pct",
                        "realized_vol_pct",
                        "recovery_boost_active",
                        "health_factor",
                    ]
                    if column in history.columns
                ]
                if visible_columns:
                    st.dataframe(history[visible_columns].tail(250), use_container_width=True)

with tabs[2]:
    if not bundle.histories or not selected_history_names:
        st.info("No local history CSVs are available for composition charts.")
    else:
        histories = _slice_histories(bundle.histories, selected_history_names, start, end)
        if not histories:
            st.info("No selected strategies have matching local history CSVs.")
            st.stop()
        prices = _load_prices()
        for name, history in histories.items():
            st.header(name)
            col_a, col_b = st.columns(2)
            collateral = build_composition_frame(history, "collateral")
            debt = build_composition_frame(history, "debt")
            with col_a:
                st.subheader("Collateral value by asset")
                if collateral.empty:
                    st.info("No collateral composition columns found.")
                else:
                    st.altair_chart(
                        _composition_chart(collateral, prices),
                        use_container_width=True,
                    )
                    st.dataframe(
                        final_composition_table(history, "collateral"),
                        use_container_width=True,
                    )
            with col_b:
                st.subheader("Debt value by asset")
                if debt.empty or float(debt.sum().sum()) == 0.0:
                    st.info("No debt composition values found.")
                else:
                    st.altair_chart(
                        _composition_chart(chart_debt_values_negative(debt), prices),
                        use_container_width=True,
                    )
                    st.dataframe(
                        final_composition_table(history, "debt"),
                        use_container_width=True,
                    )

with tabs[3]:
    if not bundle.histories or not selected_history_names:
        st.info("No local history CSVs are available for timeline charts.")
    else:
        histories = _slice_histories(bundle.histories, selected_history_names, start, end)
        if not histories:
            st.info("No selected strategies have matching local history CSVs.")
            st.stop()
        prices = _load_prices()
        plot_against_pnl = st.toggle(
            "Plot events against strategy PnL",
            value=False,
            help="When enabled, event markers use strategy PnL on the left axis and SOL price is overlaid on the right axis.",
        )
        timeline_y_mode = "strategy_pnl_usd" if plot_against_pnl else "sol_price"
        for name, history in histories.items():
            st.header(name)
            timeline = build_timeline_frame(history, prices)
            if timeline.empty:
                st.info("No timeline events could be derived for this strategy.")
                continue
            selected_families = st.multiselect(
                "Event families",
                sorted(timeline["event_family"].dropna().unique()),
                default=sorted(timeline["event_family"].dropna().unique()),
                key=f"timeline_families_{name}",
            )
            view = timeline[timeline["event_family"].isin(selected_families)]
            if view.empty:
                st.info("No events match the selected family filter.")
                continue
            st.altair_chart(
                _timeline_chart(view, y_mode=timeline_y_mode),
                use_container_width=True,
            )
            st.dataframe(
                view[
                    [
                        "timestamp",
                        "event_family",
                        "event",
                        "selected_long",
                        "target_long_fraction",
                        "target_short_fraction",
                        "health_factor",
                        "drawdown_pct",
                        "portfolio_value",
                        "strategy_pnl_usd",
                        "sol_price",
                        "portfolio_snapshot",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

with tabs[4]:
    if bundle.regimes.empty:
        st.info("No regime_summary.csv is available for this report.")
    else:
        regime_view = bundle.regimes
        if selected_names and "name" in regime_view:
            regime_view = regime_view[regime_view["name"].astype(str).isin(selected_names)]
        st.dataframe(regime_view, use_container_width=True)

with tabs[5]:
    if bundle.markdown:
        st.markdown(bundle.markdown)
    else:
        st.info("No report.md is available for this report.")

with tabs[6]:
    st.subheader("Summary")
    st.dataframe(_summary_table(summary), use_container_width=True)
    if bundle.histories:
        st.subheader("Available Histories")
        st.write(", ".join(bundle.histories.keys()))
        st.subheader("Summary to History Mapping")
        st.dataframe(
            pd.DataFrame(
                [
                    {"summary_name": name, "history_name": history_options.get(name, "")}
                    for name in strategy_names
                ]
            ),
            use_container_width=True,
        )
