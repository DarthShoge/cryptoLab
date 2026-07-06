from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "reports/latest_strategy_presets_20260627_035218"
TRANSFER = ROOT / "reports/btc_eth_directional_best_mechanics_20260628_172432"
OUT = ROOT / "src/data/strategySystemCardData.ts"

TOP_NAME = "barbell_deep70_rec1.85_dd12_gy_cd12_thr5"


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    history = pd.read_csv(path, parse_dates=["timestamp"])
    if "timestamp" in history:
        history = history.set_index("timestamp")
    return history


def _clean(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(key): _clean(value) for key, value in row.items()}
        for row in frame.to_dict("records")
    ]


def _drawdown(values: pd.Series) -> pd.Series:
    peak = values.cummax()
    return (values - peak) / peak * 100.0


def _annualized_return_pct_from_history(history: pd.DataFrame) -> float | None:
    if history.empty or "portfolio_value" not in history:
        return None
    values = pd.to_numeric(history["portfolio_value"], errors="coerce").dropna()
    if len(values) < 2:
        return None
    start_value = float(values.iloc[0])
    end_value = float(values.iloc[-1])
    if start_value <= 0 or end_value <= 0:
        return None
    days = (values.index[-1] - values.index[0]).total_seconds() / 86400.0
    if days <= 0:
        return None
    return ((end_value / start_value) ** (365.25 / days) - 1.0) * 100.0


def _annualized_return_pct_from_total(total_return_pct: Any, start: pd.Timestamp, end: pd.Timestamp) -> float | None:
    total = _clean(total_return_pct)
    if not isinstance(total, int | float):
        return None
    days = (end - start).total_seconds() / 86400.0
    if days <= 0:
        return None
    terminal_multiple = 1.0 + float(total) / 100.0
    if terminal_multiple <= 0:
        return None
    return (terminal_multiple ** (365.25 / days) - 1.0) * 100.0


def _downsample(frame: pd.DataFrame, max_points: int = 520) -> pd.DataFrame:
    if frame.empty or len(frame) <= max_points:
        return frame.copy()
    step = max(1, len(frame) // max_points)
    return frame.iloc[::step].copy()


def _series_from_history(
    history: pd.DataFrame,
    fields: list[str],
    max_points: int = 520,
) -> list[dict[str, Any]]:
    if history.empty:
        return []
    data = pd.DataFrame(index=history.index)
    for field in fields:
        if field in history:
            data[field] = pd.to_numeric(history[field], errors="coerce")
    if "portfolio_value" in history:
        values = pd.to_numeric(history["portfolio_value"], errors="coerce")
        data["drawdown_pct"] = _drawdown(values)
        data["normalized_value"] = values / float(values.dropna().iloc[0]) * 100.0
    data = _downsample(data.dropna(how="all"), max_points=max_points)
    rows = []
    for timestamp, row in data.iterrows():
        item = {"timestamp": timestamp.isoformat()}
        item.update({column: _clean(value) for column, value in row.items()})
        rows.append(item)
    return rows


def _history_path_for_latest(name: str) -> Path:
    return LATEST / f"{name}_history.csv"


def _history_path_for_transfer(name: str) -> Path:
    return TRANSFER / f"{name}_history.csv"


def _top_candidate(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {}
    match = summary[summary["name"].astype(str) == TOP_NAME]
    row = match.iloc[0] if not match.empty else summary.iloc[0]
    return {str(key): _clean(value) for key, value in row.items()}


def _with_annualized_returns(summary: pd.DataFrame, history_path_for_name: Any) -> pd.DataFrame:
    if summary.empty or "name" not in summary:
        return summary
    enriched = summary.copy()
    enriched["annualized_return_pct"] = [
        _annualized_return_pct_from_history(_load_history(history_path_for_name(str(name))))
        for name in enriched["name"].astype(str)
    ]
    return enriched


def _with_benchmark_annualized_returns(
    benchmarks: pd.DataFrame,
    reference_history: pd.DataFrame,
) -> pd.DataFrame:
    if benchmarks.empty or reference_history.empty or "total_return_pct" not in benchmarks:
        return benchmarks
    values = pd.to_numeric(reference_history.get("portfolio_value"), errors="coerce").dropna()
    if len(values) < 2:
        return benchmarks
    enriched = benchmarks.copy()
    start = values.index[0]
    end = values.index[-1]
    enriched["annualized_return_pct"] = [
        _annualized_return_pct_from_total(total_return, start, end)
        for total_return in enriched["total_return_pct"]
    ]
    return enriched


def _scenario_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    wanted = [
        "control_best_SOL_ETH",
        "best_mechanics_SOL_only_directional",
        "best_mechanics_ETH_only_directional",
        "best_mechanics_BTC_only_directional",
        "best_mechanics_BTC_ETH_directional",
    ]
    keyed = summary.assign(name=summary["name"].astype(str)).set_index("name", drop=False)
    rows = [keyed.loc[name] for name in wanted if name in keyed.index]
    return _records(pd.DataFrame(rows))


def _build_data() -> dict[str, Any]:
    latest_summary = _load_csv(LATEST / "summary.csv")
    transfer_summary = _load_csv(TRANSFER / "summary.csv")
    benchmarks = _load_csv(TRANSFER / "benchmarks.csv")
    regimes = _load_csv(TRANSFER / "regime_summary.csv")

    latest_summary = _with_annualized_returns(latest_summary, _history_path_for_latest)
    transfer_summary = _with_annualized_returns(transfer_summary, _history_path_for_transfer)

    top = _top_candidate(latest_summary)
    top_history = _load_history(_history_path_for_latest(str(top.get("name", TOP_NAME))))
    control_history = _load_history(_history_path_for_transfer("control_best_SOL_ETH"))
    sol_only_history = _load_history(_history_path_for_transfer("best_mechanics_SOL_only_directional"))
    btc_only_history = _load_history(_history_path_for_transfer("best_mechanics_BTC_only_directional"))
    eth_only_history = _load_history(_history_path_for_transfer("best_mechanics_ETH_only_directional"))
    benchmarks = _with_benchmark_annualized_returns(benchmarks, control_history)

    return {
        "meta": {
            "title": "SOL/ETH Traffic-Light Governor System Card",
            "generatedFrom": [
                str(LATEST.relative_to(ROOT)),
                str(TRANSFER.relative_to(ROOT)),
            ],
            "topCandidate": TOP_NAME,
            "window": "2021-01-01 to 2026-06-01",
        },
        "topCandidate": top,
        "latestStrategies": _records(latest_summary),
        "scenarioStrategies": _scenario_rows(transfer_summary),
        "benchmarks": _records(benchmarks),
        "regimes": _records(regimes),
        "charts": {
            "topEquity": _series_from_history(
                top_history,
                ["portfolio_value", "target_long_fraction", "target_short_fraction", "health_factor"],
            ),
            "controlEquity": _series_from_history(control_history, ["portfolio_value"]),
            "solOnlyEquity": _series_from_history(sol_only_history, ["portfolio_value"]),
            "btcOnlyEquity": _series_from_history(btc_only_history, ["portfolio_value"]),
            "ethOnlyEquity": _series_from_history(eth_only_history, ["portfolio_value"]),
        },
        "trafficStates": [
            {
                "state": "Green",
                "meaning": "Trend confirmation is broad enough to allow normal long participation.",
                "behavior": "Use traffic-light ranking to select the strongest qualifying long asset. Apply wider 5% rebalance threshold under the current turnover-control checkpoint.",
            },
            {
                "state": "Yellow",
                "meaning": "Signals are still investable but less clean.",
                "behavior": "Remain eligible for long exposure, keep cooldown active, and avoid over-trading small target changes.",
            },
            {
                "state": "Orange",
                "meaning": "Market structure is deteriorating or drawdown risk is rising.",
                "behavior": "Drawdown and volatility governors dominate sizing. The current top checkpoint does not actively short in this state.",
            },
            {
                "state": "Red",
                "meaning": "Risk state is hostile.",
                "behavior": "Exposure is cut by drawdown/volatility tiers. New long participation requires renewed traffic-light confirmation.",
            },
            {
                "state": "Recovery",
                "meaning": "Drawdown is improving and enough green confirmation has returned.",
                "behavior": "Allow re-risking up to the recovery long target of 1.85x after at least 12% drawdown, provided worsening has stopped.",
            },
        ],
        "governors": [
            "Multi-timeframe supertrend traffic-light ranking across SOL and ETH.",
            "Drawdown exposure tiers: 1.075x base, 1.125x at 30% drawdown, 0.85x at 42%, 0.70x at 50%.",
            "Realized-volatility governor with 336-hour lookback and 1.8% target, floored at 0.70x long fraction.",
            "Recovery state can re-risk to 1.85x when drawdown is improving and signal quality returns.",
            "Green/yellow 12-hour rebalance cooldown and 5% rebalance threshold to reduce churn.",
            "No same-asset debt/collateral overlap in the current implementation.",
        ],
        "useCases": [
            "Use when the mandate accepts a SOL-led risk premium and can tolerate drawdowns around the high-50% area.",
            "Use as an actively governed alternative to raw SOL buy-and-hold when reducing catastrophic drawdown matters.",
            "Use when hourly monitoring and rebalancing are operationally feasible.",
            "Use as a research checkpoint for SOL/ETH, not as a universal multi-asset policy.",
        ],
        "nonUseCases": [
            "Do not use for investors with a hard drawdown limit below 50% without further risk reduction.",
            "Do not assume the BTC-only variant works; it reduced drawdown but underperformed BTC buy-and-hold.",
            "Do not deploy without borrow availability, oracle, slippage, and health-factor monitoring.",
            "Do not treat the backtest as proof of live Kamino execution capacity.",
        ],
        "failureModes": [
            "Path dependency: sharp moves can de-risk the system before a rebound and reduce recovery capture.",
            "SOL concentration: the edge is strongly SOL-driven, so regime transfer to BTC is poor.",
            "Turnover burden: even the improved checkpoint still performs roughly 789 actions per year.",
            "Health factor proximity: minimum observed health factor is around 1.22, leaving little room for live execution slippage.",
            "Execution assumptions: fees, borrow rates, latency, oracle marks, and liquidity can materially change outcomes.",
        ],
        "productionControls": [
            "Pre-trade health-factor projection and minimum post-trade HF guard.",
            "Borrow availability and rate checks before every rebalance.",
            "Same-asset collateral/debt conflict prevention.",
            "Latency and stale-price guardrails around volatile moves.",
            "Daily reconciliation against expected holdings and debt.",
            "Kill switch for oracle anomalies, missing market data, or excessive action frequency.",
        ],
        "artifactLinks": [
            "reports/latest_strategy_presets_20260627_035218/report.md",
            "reports/latest_strategy_presets_20260627_035218/summary.csv",
            "reports/btc_eth_directional_best_mechanics_20260628_172432/report.md",
            "reports/btc_eth_directional_best_mechanics_20260628_172432/summary.csv",
            "reports/btc_eth_directional_best_mechanics_20260628_172432/benchmarks.csv",
            "reports/btc_eth_directional_best_mechanics_20260628_172432/regime_summary.csv",
        ],
    }


def main() -> None:
    data = _build_data()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "import type { StrategySystemCardData } from \"../types\";\n\n"
        "export const strategySystemCardData = "
        + json.dumps(data, indent=2, allow_nan=False)
        + " satisfies StrategySystemCardData;\n"
    )
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
