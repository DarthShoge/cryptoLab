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
    discover_report_dirs,
    final_composition_table,
    history_label_options,
    history_selection_for_summary,
    load_price_cache,
    load_report_bundle,
    regime_date_bounds,
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
        "avg_target_long_fraction",
        "avg_target_short_fraction",
        "total_interest_paid",
    ]
    existing = [column for column in columns if column in summary.columns]
    return summary[existing].copy()


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

default_index = 0
for index, path in enumerate(report_paths):
    if "strategy_comparison" in path:
        default_index = index
        break

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
default_selection = strategy_names[:2]
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

tabs = st.tabs(["Dynamics", "Composition", "Regimes", "Report", "Raw Tables"])

with tabs[0]:
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
                    st.line_chart(_downsample(history[position_columns]))
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

with tabs[1]:
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
                        _composition_chart(debt, prices),
                        use_container_width=True,
                    )
                    st.dataframe(
                        final_composition_table(history, "debt"),
                        use_container_width=True,
                    )

with tabs[2]:
    if bundle.regimes.empty:
        st.info("No regime_summary.csv is available for this report.")
    else:
        regime_view = bundle.regimes
        if selected_names and "name" in regime_view:
            regime_view = regime_view[regime_view["name"].astype(str).isin(selected_names)]
        st.dataframe(regime_view, use_container_width=True)

with tabs[3]:
    if bundle.markdown:
        st.markdown(bundle.markdown)
    else:
        st.info("No report.md is available for this report.")

with tabs[4]:
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
