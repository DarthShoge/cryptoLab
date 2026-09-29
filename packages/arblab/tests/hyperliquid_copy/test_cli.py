from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("tool",["download_hyperliquid_copy_data.py","run_hyperliquid_trader_ensemble.py"])
def test_help_offline(tool):
    result = subprocess.run([sys.executable,str(ROOT/"tools"/tool),"--help"],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_download_dry_run_no_files(tmp_path):
    result = subprocess.run([sys.executable,str(ROOT/"tools/download_hyperliquid_copy_data.py"),"pull-market",
                            "--start","2026-08-01","--end","2026-08-07","--cache-root",str(tmp_path),"--dry-run"],
                            capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []
    assert "hyperliquid_data.cli" in result.stdout


def test_smoke_rejects_validation_before_io(tmp_path):
    result = subprocess.run([sys.executable,str(ROOT/"tools/run_hyperliquid_trader_ensemble.py"),"run",
                            "--config",str(ROOT/"configs/hyperliquid_trader_ensemble_smoke.json"),
                            "--cache-root",str(tmp_path),"--study-id","smoke","--split","validation",
                            "--start","2026-08-01","--end","2026-08-08","--dry-run"],capture_output=True,text=True)
    assert result.returncode != 0 and "smoke" in result.stderr.lower()
    assert list(tmp_path.iterdir()) == []


def test_configuration_rejects_ignored_or_nonfinite_parameters():
    import json
    from arblab.hyperliquid_copy.configuration import validate_configuration
    config = json.loads((ROOT/"configs/hyperliquid_trader_ensemble_smoke.json").read_text())
    validate_configuration(config)
    with pytest.raises(ValueError,match="unknown"):
        validate_configuration(config | {"secret_strategy_parameter":1})
    with pytest.raises(ValueError):
        validate_configuration(config | {"scale_quantile":float("nan")})
    with pytest.raises(ValueError):
        validate_configuration(config | {"research_eligible":True})
