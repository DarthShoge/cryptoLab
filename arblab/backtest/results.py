"""BacktestResult container."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import pandas as pd

from arblab.backtest.market import LiquidationEvent, MarketParams
from arblab.backtest.metrics import PerformanceMetrics


@dataclass(frozen=True)
class BacktestResult:
    history: pd.DataFrame
    liquidation_events: List[LiquidationEvent]
    metrics: PerformanceMetrics
    liquidated: bool
    strategy_config: Dict[str, Any]
    engine_config: Any  # EngineConfig — avoid circular import
    market_params: MarketParams

    def summary(self) -> str:
        m = self.metrics
        lines = [
            "=== Backtest Summary ===",
            f"Return:          {m.total_return_pct:+.2f}%",
            f"Max Drawdown:    {m.max_drawdown_pct:.2f}%  ({m.max_drawdown_duration_bars} bars)",
            f"Sharpe:          {m.sharpe_ratio:.3f}",
            f"Sortino:         {m.sortino_ratio:.3f}",
            f"Min HF:          {m.min_health_factor:.3f}",
            f"Avg HF:          {m.avg_health_factor:.3f}",
            f"Bars HF<1.0:     {m.bars_below_hf_1_0}",
            f"Bars HF<1.5:     {m.bars_below_hf_1_5}",
            f"Liquidations:    {m.total_liquidations}",
            f"Interest Paid:   ${m.total_interest_paid:,.2f}",
            f"LST Yield:       ${m.total_lst_yield_earned:,.2f}",
            f"Liq Penalties:   ${m.total_liquidation_penalty_paid:,.2f}",
            f"Actions:         {m.total_actions}",
            f"Fully Liquidated: {self.liquidated}",
        ]
        return "\n".join(lines)
