import importlib.util
import json
from pathlib import Path

from test_diagnostics import diagnostic_fixture


def test_export_uses_read_only_inputs_and_new_destination(diagnostic_fixture, tmp_path):
    import pytest
    script = Path(__file__).resolve().parents[3] / "tools/export_hyperliquid_diagnostics.py"
    assert script.is_file(), "Diagnostic export helper not implemented"
    spec = importlib.util.spec_from_file_location("diagnostic_export", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    repo, item, run = diagnostic_fixture
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    output = tmp_path / "new_evidence"
    module.export(repo, item, output)
    data = json.loads((output / "diagnostics.json").read_text())
    assert data["experiment_id"] == item["id"]
    assert (output / "experiment.json").is_file()
    assert {p.name: p.read_bytes() for p in run.iterdir()} == before
    with pytest.raises(FileExistsError):
        module.export(repo, item, output)
