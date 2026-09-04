install:
    uv sync --all-packages --all-groups
    pnpm install

test:
    uv run pytest -m "not functional and not onchain and not market_data" -q

typecheck:
    pnpm --filter @cryptolab/strategy-system-card typecheck

build:
    pnpm --filter @cryptolab/strategy-system-card build

dev-kamino:
    uv run --package kamino-simulator streamlit run apps/kamino-simulator/src/kamino_simulator/app.py

dev-backtester:
    uv run --package strategy-backtester streamlit run apps/strategy-backtester/src/strategy_backtester/app.py

dev-reports:
    uv run --package report-explorer streamlit run apps/report-explorer/src/report_explorer/app.py

dev-system-card:
    pnpm --filter @cryptolab/strategy-system-card dev

generate-system-card-data:
    uv run --group tools python tools/generate_strategy_system_card_data.py

test-browser:
    uv run python -c 'from playwright.sync_api import sync_playwright; p = sync_playwright().start(); browser = p.chromium.launch(); browser.close(); p.stop()'
    uv run pytest -m functional -q
