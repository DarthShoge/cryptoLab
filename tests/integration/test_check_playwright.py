from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools/check_playwright.py"
SPEC = importlib.util.spec_from_file_location("check_playwright", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
check_playwright = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_playwright)


class FakeBrowser:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeChromium:
    def __init__(self, browser: FakeBrowser | None = None, error: Exception | None = None) -> None:
        self.browser = browser
        self.error = error

    def launch(self) -> FakeBrowser:
        if self.error is not None:
            raise self.error
        assert self.browser is not None
        return self.browser


class FakePlaywright:
    def __init__(self, chromium: FakeChromium) -> None:
        self.chromium = chromium


class FakePlaywrightContext:
    def __init__(self, playwright: FakePlaywright) -> None:
        self.playwright = playwright
        self.exited = False

    def __enter__(self) -> FakePlaywright:
        return self.playwright

    def __exit__(self, *_args: object) -> None:
        self.exited = True


def test_main_launches_and_closes_chromium(monkeypatch) -> None:
    browser = FakeBrowser()
    context = FakePlaywrightContext(FakePlaywright(FakeChromium(browser=browser)))
    monkeypatch.setattr(check_playwright, "sync_playwright", lambda: context)

    assert check_playwright.main() == 0
    assert browser.closed
    assert context.exited


def test_main_reports_actionable_launch_failure(monkeypatch, capsys) -> None:
    context = FakePlaywrightContext(
        FakePlaywright(FakeChromium(error=RuntimeError("browser executable missing")))
    )
    monkeypatch.setattr(check_playwright, "sync_playwright", lambda: context)

    assert check_playwright.main() != 0
    assert "Run just install-browser" in capsys.readouterr().err
    assert context.exited
