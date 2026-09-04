from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


GENERATOR_PATH = Path(__file__).resolve().parents[2] / "tools/generate_strategy_system_card_data.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("strategy_system_card_generator", GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _configure_valid_generator(tmp_path: Path):
    generator = _load_generator()
    generator.ROOT = tmp_path
    generator.LATEST = tmp_path / "reports" / "latest"
    generator.TRANSFER = tmp_path / "reports" / "transfer"
    generator.OUT = tmp_path / "output" / "strategySystemCardData.ts"
    generator.LATEST.mkdir(parents=True)
    generator.TRANSFER.mkdir(parents=True)
    generator.OUT.parent.mkdir(parents=True)
    generator.OUT.write_bytes(b"sentinel")

    latest_names = [generator.TOP_NAME]
    transfer_names = list(generator.TRANSFER_CHART_NAMES)
    (generator.LATEST / "summary.csv").write_text(
        "name,score\n" + "\n".join(f"{name},1" for name in latest_names) + "\n"
    )
    (generator.TRANSFER / "summary.csv").write_text(
        "name,score\n" + "\n".join(f"{name},1" for name in transfer_names) + "\n"
    )
    (generator.TRANSFER / "benchmarks.csv").write_text("name,total_return_pct\nbenchmark,10\n")
    (generator.TRANSFER / "regime_summary.csv").write_text("regime,score\ngreen,1\n")
    history = "timestamp,portfolio_value\n2025-01-01T00:00:00Z,100\n2025-01-02T00:00:00Z,101\n"
    for name in latest_names:
        (generator.LATEST / f"{name}_history.csv").write_text(history)
    for name in transfer_names:
        (generator.TRANSFER / f"{name}_history.csv").write_text(history)
    return generator


def test_main_reports_missing_inputs_without_replacing_existing_output(tmp_path: Path) -> None:
    generator = _load_generator()
    generator.ROOT = tmp_path
    generator.LATEST = tmp_path / "reports" / "latest"
    generator.TRANSFER = tmp_path / "reports" / "transfer"
    generator.OUT = tmp_path / "output" / "strategySystemCardData.ts"
    generator.OUT.parent.mkdir(parents=True)
    generator.OUT.write_bytes(b"sentinel")

    with pytest.raises(FileNotFoundError) as exc_info:
        generator.main()

    message = str(exc_info.value)
    expected_missing = [
        generator.LATEST / "summary.csv",
        generator.LATEST / f"{generator.TOP_NAME}_history.csv",
        generator.TRANSFER / "summary.csv",
        generator.TRANSFER / "benchmarks.csv",
        generator.TRANSFER / "regime_summary.csv",
        generator.TRANSFER / "control_best_SOL_ETH_history.csv",
        generator.TRANSFER / "best_mechanics_SOL_only_directional_history.csv",
        generator.TRANSFER / "best_mechanics_BTC_only_directional_history.csv",
        generator.TRANSFER / "best_mechanics_ETH_only_directional_history.csv",
    ]
    assert message.startswith("Missing required strategy system card inputs:\n")
    assert all(f"- {path}" in message for path in expected_missing)
    assert generator.OUT.read_bytes() == b"sentinel"


def test_main_atomically_replaces_output_for_valid_inputs(tmp_path: Path) -> None:
    generator = _configure_valid_generator(tmp_path)

    generator.main()

    output = generator.OUT.read_text()
    assert output.startswith('import type { StrategySystemCardData } from "../types";')
    assert output.endswith(" satisfies StrategySystemCardData;\n")
    assert list(generator.OUT.parent.glob(f".{generator.OUT.name}.*")) == []


def test_main_removes_temporary_output_when_writing_fails(tmp_path: Path, monkeypatch) -> None:
    generator = _configure_valid_generator(tmp_path)
    real_named_temporary_file = generator.tempfile.NamedTemporaryFile

    class FailingTemporaryFile:
        def __init__(self, *args, **kwargs):
            self._file = real_named_temporary_file(*args, **kwargs)
            self.name = self._file.name

        def __enter__(self):
            self._file.__enter__()
            return self

        def __exit__(self, *args):
            return self._file.__exit__(*args)

        def write(self, content: str) -> None:
            self._file.write(content[:8])
            raise OSError("simulated write failure")

    monkeypatch.setattr(generator.tempfile, "NamedTemporaryFile", FailingTemporaryFile)

    with pytest.raises(OSError, match="simulated write failure"):
        generator.main()

    assert generator.OUT.read_bytes() == b"sentinel"
    assert list(generator.OUT.parent.glob(f".{generator.OUT.name}.*")) == []
