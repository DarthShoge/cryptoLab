"""Playwright-based functional tests for the Kamino Risk Simulator Streamlit app.

These tests verify that the app starts and renders its initial UI correctly.
They do NOT require any RPC calls -- they only check the unloaded state.

Run with:  pytest tests/test_app_functional.py -m functional
"""

import subprocess
import time

import pytest
from playwright.sync_api import sync_playwright


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def app_url():
    """Start the Streamlit app on port 8502 for testing."""
    proc = subprocess.Popen(
        [
            "streamlit", "run", "kamino_app.py",
            "--server.port", "8502",
            "--server.headless", "true",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(5)  # Wait for the server to be ready
    yield "http://localhost:8502"
    proc.terminate()
    proc.wait()


@pytest.fixture(scope="module")
def browser_page(app_url):
    """Launch a headless Chromium browser and provide a page pointed at the app."""
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(app_url, wait_until="networkidle")
        yield page
        browser.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.functional
def test_initial_load(browser_page):
    """The page title should contain 'Kamino'."""
    title = browser_page.title()
    assert "Kamino" in title, f"Expected 'Kamino' in page title, got: {title}"


@pytest.mark.functional
def test_wallet_input_exists(browser_page):
    """The sidebar should contain a text input for the wallet address."""
    sidebar = browser_page.locator('[data-testid="stSidebar"]')
    wallet_input = sidebar.locator('input[type="text"]').first
    assert wallet_input.is_visible(), "Wallet text input not found in sidebar"


@pytest.mark.functional
def test_instructions_shown(browser_page):
    """An info message should prompt the user to enter an address."""
    info_text = browser_page.locator("text=Enter a wallet or obligation address")
    assert info_text.is_visible(), "Instruction text not found on initial load"


@pytest.mark.functional
def test_scenario_simulator_not_shown_without_data(browser_page):
    """The Scenario Simulator section should not appear before loading data."""
    scenario_header = browser_page.locator("text=Scenario Simulator")
    assert scenario_header.count() == 0, "Scenario Simulator should not be visible without loaded data"


@pytest.mark.functional
def test_sidebar_title(browser_page):
    """The sidebar should display the 'Kamino Risk Sim' title."""
    sidebar = browser_page.locator('[data-testid="stSidebar"]')
    title = sidebar.locator("text=Kamino Risk Sim")
    assert title.is_visible(), "Sidebar title 'Kamino Risk Sim' not found"


@pytest.mark.functional
def test_load_by_radio_buttons(browser_page):
    """The sidebar should have radio options for 'Wallet address' and 'Obligation address'."""
    wallet_radio = browser_page.locator("text=Wallet address")
    obligation_radio = browser_page.locator("text=Obligation address")
    assert wallet_radio.count() >= 1, "Wallet address radio option not found"
    assert obligation_radio.count() >= 1, "Obligation address radio option not found"


@pytest.mark.functional
def test_load_positions_button_exists(browser_page):
    """The sidebar should have a 'Load positions' button."""
    sidebar = browser_page.locator('[data-testid="stSidebar"]')
    button = sidebar.locator('button:has-text("Load positions")')
    assert button.is_visible(), "Load positions button not found in sidebar"


@pytest.mark.functional
def test_main_heading(browser_page):
    """The main area should show the 'Kamino Liquidation Risk Simulator' heading."""
    heading = browser_page.locator("text=Kamino Liquidation Risk Simulator")
    assert heading.is_visible(), "Main heading not found on page"
