# Polyglot monorepo migration baseline

Captured on 2026-09-04 before structural migration work. This record describes the
approved source-tree checkpoint and the validation baseline for later migration
tasks.

## Repository and snapshot state

- Original approved-spec SHA: `aef10192e859099b61936023e5c1e0078262d4b3`
- Original working tree: `/home/lshoge/code/cryptoLab`
- Approved tracked-work checkpoint: `64570f7e202bc8640e23fdf5af49b188f44addbc`
  (`chore: checkpoint pre-monorepo working tree`)
- Migration worktree: `/home/lshoge/code/cryptoLab/.worktrees/polyglot-monorepo`
- Preserved source snapshot: `/tmp/cryptolab-monorepo-snapshot.nNbywJ`
- The worktree was clean immediately after the checkpoint commit.

The task handoff described six modified paths, but the captured status contains
seven. The seven-path status and the content checksum below were treated as
authoritative after explicit confirmation:

- `.gitignore`
- `arblab/backtest/report_explorer.py`
- `arblab/strategies/multi_asset_traffic_light.py`
- `docs/superpowers/specs/2026-07-04-btc-pure-perp-signal-design.md`
- `strategy_report_app.py`
- `tests/test_multi_asset_traffic_light_strategy.py`
- `tests/test_report_explorer.py`

There were no staged changes in the source snapshot. The unstaged full-index patch
and the patch applied to this worktree are byte-identical:

| Snapshot artifact | SHA-256 |
| --- | --- |
| `status.z` | `fd25bfcd992b3324c8795d5f94deaabbdc095817ec183042861f7c1b34eba93f` |
| `unstaged.patch` | `7445a5126fff867449ab93822ab79e9a1acab1182836e028abc2ce4048acc2c6` |
| `staged.patch` (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `worktree-applied.patch` | `7445a5126fff867449ab93822ab79e9a1acab1182836e028abc2ce4048acc2c6` |
| `untracked.z` | `f6d455ee10f7fcd521466f530abe66c75f37a26d6ab4165123ce85f9afab63f1` |
| `untracked.sha256` | `d29b27cedb7b91f87920e6c078b6e4e2d3ddf9f7a08806d289d9a98c53b30217` |

The original tree also had 23 untracked files, retained only in the original tree
and represented by the snapshot manifest. They are summarized by directory:

- `reports/btc_eth_directional_best_mechanics_20260628_172432/` (5 files)
- `reports/full_portfolio_overview_20260627_033803/` (5 files)
- `reports/latest_strategy_presets_20260627_035218/` (3 files)
- `research/catboost-cross-sectional-crypto/` (10 files)

## Tool versions

| Tool | Version |
| --- | --- |
| uv | `0.10.0` |
| Project Python via uv | `3.12.3` |
| Node.js | `v18.19.1` |
| npm | `9.2.0` |

The shell's bare `python` command was not selected by pyenv; all Python baseline
execution therefore used the project interpreter through `uv run`.

## Baseline commands and results

| Command | Result |
| --- | --- |
| `env PYTHONDONTWRITEBYTECODE=1 uv run pytest -p no:cacheprovider -q` | PASS: 350 passed, 1 skipped in 3.76s |
| `npm run typecheck` | PASS: exit 0 (`tsc -b --pretty false`) |
| `npm run build` | PASS: exit 0; Vite 5.4.21 built 34 modules in 834ms |

The first sandboxed pytest attempt exited 2 before collection because uv could not
write `/home/lshoge/.cache/uv/sdists-v9/.git`. Re-running the exact command with
authorized access to the existing uv cache passed. This was an execution-sandbox
failure, not a repository test failure.

The frontend build emitted its existing advisory that the generated 707.60 kB JS
chunk exceeds Vite's 500 kB warning threshold. It did not fail the build.

## Opt-in and external behavior

- `tests/test_app_functional.py` contains eight Playwright browser tests marked
  `functional`. The module uses `pytest.importorskip("playwright")`; Playwright was
  unavailable in this baseline environment, accounting for the single skipped
  module. Running these tests also requires a Chromium installation and starts a
  local Streamlit server.
- Scenario and backtest markers are part of the current offline suite and were not
  excluded from the successful baseline command.
- No tests are currently marked `onchain` or `market_data`. Unit tests for on-chain
  decoding and market-data access use local inputs or mocks. Live Solana RPC and
  remote exchange calls remain runtime/explicit behaviors and were not exercised.
- `legacy/exchange-arbitrage/main.py` was not executed.

There are no pre-existing test, typecheck, or build failures to carry forward. The
uv sandbox initialization error and Vite chunk-size advisory above are the only
observed non-success signatures.
