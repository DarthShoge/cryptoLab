# CryptoLab

CryptoLab is a polyglot workspace for DeFi lending-risk analysis and strategy research. It contains four independently runnable frontends:

- `kamino-simulator`: Streamlit liquidation-risk simulator and CLI
- `strategy-backtester`: Streamlit historical strategy backtester
- `report-explorer`: Streamlit explorer for generated research reports
- `strategy-system-card`: React/Vite strategy system card

The three Python frontends share the `arblab` package. The JavaScript frontend is an independent pnpm workspace package.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/)
- Node.js 18 or newer (required by the pinned Playwright 1.58.2 toolchain)
- pnpm 10.30.3, matching the root `packageManager` field; use Corepack when available or the [official pnpm installation](https://pnpm.io/installation)
- [just](https://just.systems/)

## Setup

```bash
just install
cp .env.example .env
```

`just install` runs the locked Python and JavaScript workspace installs. Edit `.env` if you need non-default RPC configuration. Chromium is not part of normal setup; install it only before browser functional tests:

```bash
just install-browser
```

## Run the frontends

Each frontend runs independently from the repository root:

```bash
just dev-kamino
just dev-backtester
just dev-reports
just dev-system-card
```

The direct equivalents are:

```bash
uv run --package kamino-simulator streamlit run apps/kamino-simulator/src/kamino_simulator/app.py
uv run --package strategy-backtester streamlit run apps/strategy-backtester/src/strategy_backtester/app.py
uv run --package report-explorer streamlit run apps/report-explorer/src/report_explorer/app.py
pnpm --filter @cryptolab/strategy-system-card dev
```

## Kamino CLI

With no arguments, the CLI runs the bundled offline sample:

```bash
uv run --package kamino-simulator python -m kamino_simulator.cli
```

Use an explicit snapshot file, or replace `YOUR_OBLIGATION_ADDRESS` with an obligation address to load it from RPC using the bundled Kamino IDL:

```bash
uv run --package kamino-simulator python -m kamino_simulator.cli \
  --input data/fixtures/kamino_sample.json

uv run --package kamino-simulator python -m kamino_simulator.cli \
  --obligation YOUR_OBLIGATION_ADDRESS \
  --idl data/fixtures/kamino_idl.json
```

The custom strategy API remains importable from the shared package:

```python
from arblab.backtest.strategy import Strategy

class MyStrategy(Strategy):
    def setup(self, snapshot, config):
        return snapshot

    def on_bar(self, snapshot, bar):
        return []

    def on_liquidation(self, snapshot, bar, event):
        return []
```

## Tests

The default suite is offline and excludes browser, Solana RPC, and remote market-data tests:

```bash
just test
# direct equivalent
uv run pytest -q
```

Focused examples use the workspace's current test locations:

```bash
uv run pytest packages/arblab/tests -q
uv run pytest apps/kamino-simulator/tests -q
uv run pytest apps/strategy-backtester/tests -q
uv run pytest apps/report-explorer/tests -q
uv run pytest tests/integration -q
uv run pytest -m scenario -q
```

Browser tests are explicit and require the separate browser installation:

```bash
just install-browser
just test-browser
```

Tests marked `onchain` (Solana RPC) or `market_data` (remote market data) are opt-in when such tests exist, for example `uv run pytest -m onchain` or `uv run pytest -m market_data`.

## Hyperliquid trader ensemble

The [offline trader-ensemble prototype](docs/hyperliquid-trader-ensemble.md) reuses
Hyperliquid data helpers and tests causal wallet ranking, ensemble signals and
delayed perpetual execution. It cannot submit exchange orders. Paid archive
downloads require explicit cost acceptance; research and live-paper qualification
remain separate checkpoints.

## Checks and generated data

```bash
just typecheck
just build
just generate-system-card-data
```

Direct equivalents:

```bash
pnpm --filter @cryptolab/strategy-system-card typecheck
pnpm --filter @cryptolab/strategy-system-card build
uv run --group tools python tools/generate_strategy_system_card_data.py
```

The system-card generator reads local report source artifacts under `reports/`. It validates all required inputs before replacing generated data and fails safely without changing the output when those artifacts are absent.

## Repository layout and ownership

```text
apps/
  kamino-simulator/       Python Streamlit UI, CLI, and app tests
  strategy-backtester/    Python Streamlit UI and app tests
  report-explorer/        Python Streamlit report UI and app tests
  strategy-system-card/   React/Vite UI and frontend-owned source
packages/
  arblab/                 Shared Python domain, backtest, and strategy library
tests/integration/        Cross-workspace integration checks
tools/                    Workspace generators and browser preflight tooling
data/fixtures/            Checked-in sample inputs and Kamino IDL
docs/                     Current architecture, strategy, and migration docs
reports/                  Generated or captured strategy results
research/                 Research sources and synthesis
notebooks/                Exploratory analysis
legacy/                   Isolated superseded experiments; not workspace code
```

Code under `apps/` owns frontend entrypoints and app-specific tests; reusable domain and backtest behavior belongs in `packages/arblab`. Treat `data/`, `reports/`, `research/`, and `notebooks/` as project artifacts rather than application packages. The `legacy/` tree is retained for provenance and its commands and dependency metadata are not valid workspace setup instructions.
