# Multi-Asset Traffic-Light Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the first curated-universe multi-asset traffic-light research experiment.

**Architecture:** Add reusable signal scoring, a conservative dynamic long/short Kamino research strategy, and a timestamped sweep/report script. Keep the existing SOL strategy unchanged and compare results against the current checkpoint.

**Tech Stack:** Python, pandas, pytest, existing `BacktestEngine`, `MarketParams`, `OHLCVConfig`, and Kamino action model.

---

### Task 1: Multi-Asset Signal Helpers

**Files:**
- Create: `arblab/backtest/traffic_lights.py`
- Test: `tests/test_backtest_traffic_lights.py`

- [ ] Write failing tests for per-asset multi-timeframe signal scoring.
- [ ] Run `uv run pytest tests/test_backtest_traffic_lights.py` and verify failure.
- [ ] Implement signal helpers using the existing `supertrend_direction`.
- [ ] Run `uv run pytest tests/test_backtest_traffic_lights.py`.

### Task 2: Dynamic Pair Strategy

**Files:**
- Create: `arblab/strategies/multi_asset_traffic_light.py`
- Test: `tests/test_multi_asset_traffic_light_strategy.py`

- [ ] Write failing tests for setup, long/short pair selection, and de-risk behavior.
- [ ] Run `uv run pytest tests/test_multi_asset_traffic_light_strategy.py` and verify failure.
- [ ] Implement a conservative strategy that rebalances between USDC, the strongest long asset, and the weakest short asset.
- [ ] Run focused strategy tests.

### Task 3: Research Runner And Report

**Files:**
- Create: `temp/run_multi_asset_traffic_light_sweep.py`
- Create report artifacts under `reports/multi_asset_traffic_light_YYYYMMDD_HHMMSS/`

- [ ] Write a small sweep across signal thresholds and target short ratios.
- [ ] Run focused tests for changed modules.
- [ ] Run the sweep using cached data where possible.
- [ ] Write a report that compares candidates against the SOL checkpoint.

### Task 4: Commit

- [ ] Run `uv run pytest`.
- [ ] Stage only relevant files.
- [ ] Commit with message `research: test multi asset traffic light selector`.
