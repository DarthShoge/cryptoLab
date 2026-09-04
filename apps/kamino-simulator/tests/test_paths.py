"""Packaging and startup tests for the Kamino simulator app."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from arblab.paths import fixture_path


APP_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = APP_ROOT / "src" / "kamino_simulator" / "app.py"
CLI_PATH = APP_ROOT / "src" / "kamino_simulator" / "cli.py"


def test_app_and_cli_live_in_importable_package() -> None:
    assert APP_PATH.is_file()
    assert CLI_PATH.is_file()


def test_default_idl_path_is_the_stable_fixture_from_any_cwd(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import streamlit as st

    class InitialRenderStopped(BaseException):
        pass

    def stop_initial_render() -> None:
        raise InitialRenderStopped

    monkeypatch.setattr(st, "stop", stop_initial_render)
    spec = importlib.util.spec_from_file_location("kamino_simulator.app_path_test", APP_PATH)
    assert spec is not None
    assert spec.loader is not None
    app = importlib.util.module_from_spec(spec)
    with pytest.raises(InitialRenderStopped):
        spec.loader.exec_module(app)
    expected = fixture_path("kamino_idl.json")

    assert app.IDL_PATH == expected
    assert app.IDL_PATH.is_absolute()
    assert app.IDL_PATH.is_file()

    monkeypatch.chdir(tmp_path)

    assert app.IDL_PATH == fixture_path("kamino_idl.json")
    assert app.IDL_PATH.is_absolute()
    assert app.IDL_PATH.is_file()


def test_initial_streamlit_load_does_not_call_network_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from streamlit.testing.v1 import AppTest

    import arblab.kamino_onchain as onchain

    def unexpected_network_call(*args: object, **kwargs: object) -> None:
        pytest.fail("initial app load called an on-chain or network helper")

    for name in (
        "_fetch_jupiter_symbols",
        "find_wallet_obligations",
        "get_obligation_market",
        "load_market_reserves",
        "load_onchain_snapshot",
    ):
        monkeypatch.setattr(onchain, name, unexpected_network_call)

    app = AppTest.from_file(str(APP_PATH)).run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Kamino Liquidation Risk Simulator"
