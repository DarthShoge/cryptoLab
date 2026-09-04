"""Packaging and startup tests for the strategy backtester app."""

from __future__ import annotations

import ast
import socket
from pathlib import Path

import pytest


APP_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = APP_ROOT / "src" / "strategy_backtester" / "app.py"


def test_app_lives_in_importable_package_and_uses_local_helpers() -> None:
    assert APP_PATH.is_file()

    imports = {
        (node.level, node.module)
        for node in ast.walk(ast.parse(APP_PATH.read_text()))
        if isinstance(node, ast.ImportFrom)
    }

    assert (0, "strategy_backtester.app_helpers") in imports or (
        1,
        "app_helpers",
    ) in imports
    assert (0, "arblab.backtest.app_helpers") not in imports


def test_initial_streamlit_load_does_not_fetch_market_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest

    import arblab.backtest.data as market_data

    def unexpected_market_data_call(*args: object, **kwargs: object) -> None:
        pytest.fail("initial app load requested market data")

    def unexpected_network_call(*args: object, **kwargs: object) -> None:
        raise AssertionError("initial app load attempted an outbound connection")

    monkeypatch.setattr(market_data, "fetch_ohlcv", unexpected_market_data_call)
    monkeypatch.setattr(socket, "create_connection", unexpected_network_call)
    monkeypatch.setattr(socket.socket, "connect", unexpected_network_call)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Kamino Lending Strategy Backtester"
