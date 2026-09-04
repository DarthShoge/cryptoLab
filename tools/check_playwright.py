from __future__ import annotations

import sys

from playwright.sync_api import sync_playwright


def main() -> int:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            try:
                return 0
            finally:
                browser.close()
    except Exception as error:
        print(
            f"Playwright Chromium preflight failed: {error}\n"
            "Run just install-browser to install the required browser.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
