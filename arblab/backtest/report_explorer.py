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
