"""Command-line behavior for the Kamino simulator."""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

import pytest

from kamino_simulator import cli
from arblab.kamino_risk import AccountSnapshot
from arblab.paths import fixture_path


def _run_cli(monkeypatch: pytest.MonkeyPatch, *args: str) -> None:
    monkeypatch.setattr(sys, "argv", ["kamino-simulator", *args])
    cli.main()


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


def test_input_and_obligation_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["kamino-simulator", "--input", "sample.json", "--obligation", "address"],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "not allowed with argument" in capsys.readouterr().err


@pytest.mark.parametrize(
    "flags",
    [
        ("--idl", "idl.json"),
        ("--rpc-url", "http://localhost:8899"),
        ("--program-id", "program"),
        ("--obligation-account-name", "CustomObligation"),
        ("--reserve-account-name", "CustomReserve"),
    ],
)
def test_online_flags_require_obligation(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    flags: tuple[str, str],
) -> None:
    monkeypatch.setattr(sys, "argv", ["kamino-simulator", *flags])

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "require --obligation" in capsys.readouterr().err


def test_input_rejects_online_only_flags(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kamino-simulator",
            "--input",
            str(fixture_path("kamino_sample.json")),
            "--idl",
            "idl.json",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    assert "cannot be used with --input" in capsys.readouterr().err


def test_obligation_without_idl_reports_only_missing_requirement(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["kamino-simulator", "--obligation", "address"])

    with pytest.raises(SystemExit, match="2"):
        cli.main()

    error = capsys.readouterr().err
    assert "--idl is required with --obligation" in error
    assert "--program-id" not in error.splitlines()[-1]


def test_explicit_input_still_runs_offline(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        cli,
        "load_onchain_snapshot",
        lambda *args, **kwargs: pytest.fail("input mode called the on-chain loader"),
    )

    _run_cli(monkeypatch, "--input", str(fixture_path("kamino_sample.json")))

    assert "Baseline:" in capsys.readouterr().out


def test_obligation_uses_environment_program_and_rpc_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def capture_loader(**kwargs: object) -> AccountSnapshot:
        captured.update(kwargs)
        return AccountSnapshot(collateral=[], debt=[])

    monkeypatch.setenv("KAMINO_PROGRAM_ID", "env-program")
    monkeypatch.setenv("SOLANA_RPC_URL", "http://env-rpc")
    monkeypatch.setattr(cli, "load_onchain_snapshot", capture_loader)
    idl_path = tmp_path / "idl.json"

    _run_cli(monkeypatch, "--obligation", "address", "--idl", str(idl_path))

    assert captured["program_id"] == "env-program"
    assert captured["rpc_url"] == "http://env-rpc"


def test_obligation_uses_mainnet_defaults_when_environment_is_unset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def capture_loader(**kwargs: object) -> AccountSnapshot:
        captured.update(kwargs)
        return AccountSnapshot(collateral=[], debt=[])

    monkeypatch.delenv("KAMINO_PROGRAM_ID", raising=False)
    monkeypatch.delenv("SOLANA_RPC_URL", raising=False)
    monkeypatch.setattr(cli, "load_onchain_snapshot", capture_loader)

    _run_cli(monkeypatch, "--obligation", "address", "--idl", str(tmp_path / "idl.json"))

    assert captured["program_id"] == "KLend2g3cP87fffoy8q1mQqGKjrxjC8boSyAYavgmjD"
    assert captured["rpc_url"] == "https://api.mainnet-beta.solana.com"


def test_no_arguments_subprocess_runs_sample_from_unrelated_cwd(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "kamino_simulator.cli"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Baseline:" in result.stdout
    assert "baseline_liquidation_prices" in result.stdout
