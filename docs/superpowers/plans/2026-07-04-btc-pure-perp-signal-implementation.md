# BTC Pure Perp Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a BTC-only pure signal strategy that emits `[-1, 1]` target perp exposure and backtests it with Binance-style perp execution, fees, stop loss, and take profit controls.

**Architecture:** Add a new focused perp research package under `arblab/perps/`. Keep signal generation stateless and separate from the path-dependent perp account simulator. Add a report runner that writes timestamped artifacts without modifying existing Kamino backtest abstractions.

**Tech Stack:** Python 3.12, pandas, existing `uv run pytest`, existing `arblab.backtest.data.fetch_ohlcv`, existing `supertrend_direction` helper.

---

## File Structure

- Create `arblab/perps/__init__.py`: package exports.
- Create `arblab/perps/signal.py`: indicator helpers and BTC pure signal generation.
- Create `arblab/perps/simulator.py`: single-market perp account simulator with fees, deadband, stop loss, take profit, and re-entry guard.
- Create `arblab/perps/report.py`: summary metrics and artifact writing.
- Create `tools/run_btc_perp_signal_backtest.py`: CLI runner for Binance BTCUSDT research runs.
- Create `tests/test_perp_signal.py`: signal and closed-candle tests.
- Create `tests/test_perp_simulator.py`: accounting, deadband, stop loss, take profit, and re-entry tests.
- Create `tests/test_perp_report.py`: summary and artifact smoke tests.

## Task 1: Signal Module

**Files:**
- Create: `arblab/perps/__init__.py`
- Create: `arblab/perps/signal.py`
- Test: `tests/test_perp_signal.py`

- [ ] **Step 1: Write failing tests for signal bounds, no-trade zone, and explainable components**

Run: `uv run pytest tests/test_perp_signal.py -v`

Expected: FAIL because `arblab.perps.signal` does not exist.

- [ ] **Step 2: Implement minimal signal dataclasses and weighted signal calculation**

Implement `PureSignalConfig`, `generate_btc_signal(price_data, config)`, EMA vote, RSI modifier, weighted blend, clamp, and no-trade zone.

- [ ] **Step 3: Run signal tests**

Run: `uv run pytest tests/test_perp_signal.py -v`

Expected: PASS.

- [ ] **Step 4: Add closed-candle higher-timeframe test**

Test that a higher-timeframe Supertrend vote is shifted before forward-fill so the current in-progress higher-timeframe candle cannot affect the current base bar.

- [ ] **Step 5: Implement or adjust closed-candle timeframe mapping**

Reuse the existing traffic-light resampling pattern where practical.

- [ ] **Step 6: Run signal tests again**

Run: `uv run pytest tests/test_perp_signal.py -v`

Expected: PASS.

## Task 2: Perp Simulator

**Files:**
- Create: `arblab/perps/simulator.py`
- Test: `tests/test_perp_simulator.py`

- [ ] **Step 1: Write failing tests for deterministic long PnL and fee accounting**

Use a tiny price/signal frame where BTC rises from 100 to 110 with `1.0x` exposure and known fee rate.

Run: `uv run pytest tests/test_perp_simulator.py -v`

Expected: FAIL because simulator does not exist.

- [ ] **Step 2: Implement simulator config, state, and basic mark-to-market loop**

Implement `PerpSimulatorConfig` and `simulate_perp_account(price_data, signals, config)`.

- [ ] **Step 3: Run simulator tests**

Run: `uv run pytest tests/test_perp_simulator.py -v`

Expected: PASS for basic accounting.

- [ ] **Step 4: Add failing tests for rebalance deadband**

Verify small target changes do not trade.

- [ ] **Step 5: Implement deadband behavior**

Apply deadband to exposure changes after stop/take-profit and re-entry logic.

- [ ] **Step 6: Add failing tests for fixed percent stop loss and take profit on long and short positions**

Use deterministic price paths and assert exit reason, flat exposure, and stop/take-profit counts.

- [ ] **Step 7: Implement fixed percent stop loss and take profit**

Track average entry. On same-direction adds, update notional-weighted average entry. On reductions, keep existing average entry. On flips, close and reopen with a fresh entry.

- [ ] **Step 8: Add failing test for same-direction re-entry guard**

After SL/TP exit, verify the simulator blocks same-direction re-entry until the raw signal goes flat or flips direction.

- [ ] **Step 9: Implement re-entry guard**

Track blocked direction after SL/TP. Clear it when signal is flat or opposite.

- [ ] **Step 10: Run simulator tests**

Run: `uv run pytest tests/test_perp_simulator.py -v`

Expected: PASS.

## Task 3: Report Artifacts

**Files:**
- Create: `arblab/perps/report.py`
- Test: `tests/test_perp_report.py`

- [ ] **Step 1: Write failing tests for summary fields and artifact files**

Expected files: `signals.csv`, `history.csv`, `summary.json`, `report.md`.

Run: `uv run pytest tests/test_perp_report.py -v`

Expected: FAIL because report module does not exist.

- [ ] **Step 2: Implement summary metrics and artifact writer**

Include total return, annualized return, max drawdown, Sharpe, Sortino, turnover, fee drag, funding drag, stop count, take-profit count, percent time long, percent time short, percent time flat, and BTC benchmark return.

- [ ] **Step 3: Run report tests**

Run: `uv run pytest tests/test_perp_report.py -v`

Expected: PASS.

## Task 4: CLI Runner

**Files:**
- Create: `tools/run_btc_perp_signal_backtest.py`
- Test: focused smoke via pytest/import and optional script help command.

- [ ] **Step 1: Write failing import/smoke test if existing test patterns support tools imports**

If no tool test pattern exists, keep verification to direct `--help` execution.

- [ ] **Step 2: Implement CLI runner**

Fetch Binance `BTC/USDT` OHLCV with existing cache helper, generate signals, simulate the account, and write artifacts to `reports/btc_pure_perp_signal_<timestamp>/`.

- [ ] **Step 3: Run CLI help**

Run: `uv run python tools/run_btc_perp_signal_backtest.py --help`

Expected: exit 0 with option descriptions.

## Task 5: Final Verification

**Files:**
- All new files above.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_perp_signal.py tests/test_perp_simulator.py tests/test_perp_report.py -v`

Expected: PASS.

- [ ] **Step 2: Run full test suite**

Run: `uv run pytest`

Expected: PASS.

- [ ] **Step 3: Review git diff**

Run: `git diff --stat` and `git diff --check`

Expected: new scoped modules/tests only and no whitespace errors.

- [ ] **Step 4: Commit implementation**

Commit only the implementation files from this worktree.
