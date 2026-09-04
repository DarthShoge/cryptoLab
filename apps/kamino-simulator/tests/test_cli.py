"""Command-line behavior for the Kamino simulator."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from kamino_simulator import cli


def test_no_arguments_runs_bundled_sample_offline_from_any_cwd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_network_call(*args: object, **kwargs: object) -> None:
        pytest.fail("default CLI execution called the on-chain loader")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["kamino-simulator"])
    monkeypatch.setattr(cli, "load_onchain_snapshot", unexpected_network_call)

    cli.main()

    output = capsys.readouterr().out
    assert "Baseline:" in output
    assert "baseline_liquidation_prices" in output
