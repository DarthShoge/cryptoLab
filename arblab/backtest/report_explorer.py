"""Reusable helpers for exploring strategy report artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReportBundle:
    """Loaded strategy report artifacts."""

    path: Path
    summary: pd.DataFrame
    regimes: pd.DataFrame
    markdown: str
    histories: dict[str, pd.DataFrame]


def discover_report_dirs(root: Path = Path("reports")) -> list[Path]:
    """Return report directories that contain a report or summary artifact."""
    if not root.exists():
        return []
    dirs = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and ((path / "summary.csv").exists() or (path / "report.md").exists())
    ]
    return sorted(dirs, key=lambda path: path.name, reverse=True)


def load_report_bundle(path: Path) -> ReportBundle:
    """Load summary, regime, markdown, and local history CSVs for a report."""
    summary = _read_csv_if_exists(path / "summary.csv")
    regimes = _read_csv_if_exists(path / "regime_summary.csv")
    markdown = (path / "report.md").read_text() if (path / "report.md").exists() else ""
    histories = _load_histories(path)
    return ReportBundle(
        path=path,
        summary=summary,
        regimes=regimes,
        markdown=markdown,
        histories=histories,
    )


def max_drawdown_series(values: pd.Series) -> pd.Series:
    """Return positive drawdown percentage for a portfolio value series."""
    if values.empty:
        return pd.Series(dtype=float, index=values.index)
    peak = values.cummax()
    drawdown = (values - peak) / peak
    return drawdown.abs() * 100.0


def slice_regime(
    history: pd.DataFrame,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> pd.DataFrame:
    """Slice a history DataFrame by optional inclusive start/end timestamps."""
    if history.empty:
        return history
    out = history
    if start is not None:
        out = out.loc[out.index >= _compatible_timestamp(start, out.index)]
    if end is not None:
        out = out.loc[out.index <= _compatible_timestamp(end, out.index)]
    return out


def build_metric_frame(
    histories: dict[str, pd.DataFrame],
    metric: str,
) -> pd.DataFrame:
    """Build a side-by-side chart frame for a metric across histories."""
    data: dict[str, pd.Series] = {}
    for name, history in histories.items():
        if history.empty:
            continue
        if metric == "normalized_portfolio_value":
            values = history["portfolio_value"]
            data[name] = values / float(values.iloc[0]) * 100.0
        elif metric == "drawdown_pct":
            data[name] = max_drawdown_series(history["portfolio_value"])
        elif metric in history.columns:
            data[name] = history[metric]
    return pd.DataFrame(data)


def load_price_cache(
    cache_dir: Path = Path("notebooks/.price_cache"),
    symbols: list[str] | None = None,
) -> dict[str, pd.Series]:
    """Load cached close-price series keyed by display symbol."""
    if symbols is None:
        symbols = ["SOL", "ETH"]
    if not cache_dir.exists():
        return {}

    prices: dict[str, pd.Series] = {}
    for symbol in symbols:
        matches = sorted(cache_dir.glob(f"{symbol}_USDT_1h_*.csv"))
        if not matches:
            continue
        df = pd.read_csv(matches[-1], parse_dates=["timestamp"])
        if "close" not in df:
            continue
        series = df.set_index("timestamp")["close"].astype(float)
        if series.index.tz is None:
            series.index = series.index.tz_localize("UTC")
        prices[symbol] = series
    return prices


def build_buy_hold_frame(
    histories: dict[str, pd.DataFrame],
    prices: dict[str, pd.Series],
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Build SOL/ETH buy-and-hold value series using common initial equity."""
    if symbols is None:
        symbols = ["SOL", "ETH"]
    if not histories:
        return pd.DataFrame()

    first_history = next((history for history in histories.values() if not history.empty), None)
    if first_history is None or "portfolio_value" not in first_history:
        return pd.DataFrame()

    index = first_history.index
    initial_value = float(first_history["portfolio_value"].iloc[0])
    data: dict[str, pd.Series] = {}
    for symbol in symbols:
        if symbol not in prices:
            continue
        aligned = prices[symbol].reindex(index, method="ffill")
        if aligned.empty or pd.isna(aligned.iloc[0]) or float(aligned.iloc[0]) <= 0.0:
            continue
        units = initial_value / float(aligned.iloc[0])
        data[f"Buy & Hold {symbol}"] = aligned * units
    return pd.DataFrame(data, index=index)


def build_temperature_frame(
    history: pd.DataFrame,
    prices: dict[str, pd.Series],
) -> pd.DataFrame:
    """Return aligned SOL price and net long/short exposure temperature."""
    columns = ["sol_price", "net_temperature"]
    if history.empty or "SOL" not in prices:
        return pd.DataFrame(columns=columns, index=history.index)

    long = history.get("target_long_fraction", pd.Series(0.0, index=history.index))
    short = history.get("target_short_fraction", pd.Series(0.0, index=history.index))
    sol_price = prices["SOL"].reindex(history.index, method="ffill").astype(float)
    frame = pd.DataFrame(
        {
            "sol_price": sol_price,
            "net_temperature": long.astype(float) - short.astype(float),
        },
        index=history.index,
    )
    return frame.dropna(how="any")


def build_composition_frame(history: pd.DataFrame, side: str) -> pd.DataFrame:
    """Return asset value composition for collateral or debt."""
    prefix = f"{side}_"
    data: dict[str, pd.Series] = {}
    for column in history.columns:
        if not column.startswith(prefix) or not column.endswith("_value"):
            continue
        if column in {"collateral_value", "debt_value"}:
            continue
        asset = column.removeprefix(prefix).removesuffix("_value")
        data[asset] = history[column].clip(lower=0.0)
    return pd.DataFrame(data, index=history.index)


def final_composition_table(history: pd.DataFrame, side: str) -> pd.DataFrame:
    """Return latest asset composition values and percentage shares."""
    columns = ["asset", "value_usd", "share_pct"]
    frame = build_composition_frame(history, side)
    if frame.empty:
        return pd.DataFrame(columns=columns)
    latest = frame.iloc[-1]
    total = float(latest.sum())
    rows = []
    for asset, value in latest.items():
        value = float(value)
        if value <= 0.0:
            continue
        rows.append(
            {
                "asset": asset,
                "value_usd": value,
                "share_pct": value / total * 100.0 if total > 0.0 else 0.0,
            }
        )
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).sort_values("value_usd", ascending=False).reset_index(drop=True)


def history_label_options(
    summary: pd.DataFrame,
    histories: dict[str, pd.DataFrame],
) -> dict[str, str]:
    """Map summary strategy labels to available history artifact labels."""
    if summary.empty or "name" not in summary:
        return {name: name for name in histories}

    summary_names = summary["name"].dropna().astype(str).tolist()
    history_names = list(histories)
    options: dict[str, str] = {}
    unused = list(history_names)
    for name in summary_names:
        match = _history_match_for_summary_name(name, unused)
        if match is None:
            match = _history_match_for_summary_name(name, history_names)
        if match is not None:
            options[name] = match
            if match in unused:
                unused.remove(match)
    for name in history_names:
        options.setdefault(name, name)
    return options


def history_selection_for_summary(
    selected_summary_names: list[str],
    options: dict[str, str],
) -> list[str]:
    """Return history labels corresponding to selected summary labels."""
    selected: list[str] = []
    for name in selected_summary_names:
        history_name = options.get(name, name)
        if history_name not in selected:
            selected.append(history_name)
    return selected


def regime_date_bounds(regimes: pd.DataFrame) -> dict[str, tuple[str | None, str | None]]:
    """Return known date bounds for common regime labels."""
    known = {
        "full_2021_2025": (None, None),
        "bull_2021": ("2021-01-01", "2021-12-31 23:59:59+00:00"),
        "crash_2022": ("2022-01-01", "2022-12-31 23:59:59+00:00"),
        "recovery_2023": ("2023-01-01", "2023-12-31 23:59:59+00:00"),
        "post_2024": ("2024-01-01", None),
    }
    if regimes.empty or "regime" not in regimes:
        return {"full": (None, None)}
    bounds = {"full": (None, None)}
    for regime in regimes["regime"].dropna().astype(str).unique():
        bounds[regime] = known.get(regime, (None, None))
    return bounds


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _compatible_timestamp(value: str | pd.Timestamp, index: pd.Index) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    index_tz = getattr(index, "tz", None)
    if index_tz is not None and timestamp.tzinfo is None:
        return timestamp.tz_localize(index_tz)
    if index_tz is not None:
        return timestamp.tz_convert(index_tz)
    if timestamp.tzinfo is not None:
        return timestamp.tz_localize(None)
    return timestamp


def _load_histories(path: Path) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(path.glob("*history.csv")):
        history = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        if history.index.tz is None:
            history.index = history.index.tz_localize("UTC")
        name = csv_path.stem.removesuffix("_history")
        histories[name] = history
    return histories


def _history_match_for_summary_name(
    summary_name: str,
    history_names: list[str],
) -> str | None:
    if summary_name in history_names:
        return summary_name
    lowered = summary_name.lower()
    if "static" in lowered or "highest_return" in lowered:
        if "highest_return" in history_names:
            return "highest_return"
    if "sortino" in lowered or lowered.startswith("rv"):
        if "high_sortino" in history_names:
            return "high_sortino"
    compact = lowered.replace("_", "").replace(".", "")
    for history_name in history_names:
        history_compact = history_name.lower().replace("_", "").replace(".", "")
        if history_compact in compact or compact in history_compact:
            return history_name
    return None
