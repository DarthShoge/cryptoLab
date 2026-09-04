install:
    uv sync --locked --all-packages --all-groups
    pnpm install --frozen-lockfile

test:
    uv run pytest -q

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

install-browser:
    uv run playwright install chromium

test-browser:
    uv run python tools/check_playwright.py
    uv run pytest -m functional -q
