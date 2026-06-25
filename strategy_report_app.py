"""Reusable Streamlit explorer for strategy report artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from arblab.backtest.report_explorer import (
    build_metric_frame,
    discover_report_dirs,
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
history_names = [name for name in strategy_names if name in bundle.histories]

st.sidebar.header("Strategies")
default_selection = history_names[:2] if history_names else strategy_names[:2]
selected_names = st.sidebar.multiselect(
    "Compare",
    strategy_names,
    default=default_selection,
)

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

tabs = st.tabs(["Dynamics", "Regimes", "Report", "Raw Tables"])

with tabs[0]:
    if not bundle.histories or not selected_names:
        st.info("No local history CSVs are available for dynamic charts.")
    else:
        histories = _slice_histories(bundle.histories, selected_names, start, end)
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
    if bundle.regimes.empty:
        st.info("No regime_summary.csv is available for this report.")
    else:
        regime_view = bundle.regimes
        if selected_names and "name" in regime_view:
            regime_view = regime_view[regime_view["name"].astype(str).isin(selected_names)]
        st.dataframe(regime_view, use_container_width=True)

with tabs[2]:
    if bundle.markdown:
        st.markdown(bundle.markdown)
    else:
        st.info("No report.md is available for this report.")

with tabs[3]:
    st.subheader("Summary")
    st.dataframe(_summary_table(summary), use_container_width=True)
    if bundle.histories:
        st.subheader("Available Histories")
        st.write(", ".join(bundle.histories.keys()))
