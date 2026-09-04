from datetime import timedelta

import pytest

from .test_market_data import T


def test_registry_state_guards_and_idempotence(tmp_path):
    from arblab.hyperliquid_copy.trial_registry import TrialRegistry
    registry = TrialRegistry(tmp_path/"trials.jsonl")
    args = dict(study_id="test", dataset_hash="data", config_hash="cfg", split="development", smoke=False)
    trial = registry.begin(**args)
    assert registry.begin(**args) == trial
    registry.complete(trial, dict(sharpe=1, total_return=.1, max_drawdown=.1))
    assert registry.select("test","data") == "cfg"
    assert registry.select("test","data") == "cfg"
    with pytest.raises(ValueError, match="selected"):
        registry.begin(**(args | {"config_hash":"different", "split":"validation"}))
    with pytest.raises(ValueError, match="validation"):
        registry.begin(**(args | {"split":"locked-test"}), unlock="cfg")
    validation = registry.begin(**(args | {"split":"validation"}))
    registry.complete(validation, dict(sharpe=1, total_return=.05, max_drawdown=.1))
    with pytest.raises(ValueError, match="unlock"):
        registry.begin(**(args | {"split":"locked-test"}))
    locked = registry.begin(**(args | {"split":"locked-test"}), unlock="cfg")
    registry.complete(locked, dict(sharpe=1,total_return=.02,max_drawdown=.1))
    with pytest.raises(ValueError, match="once"):
        registry.begin(**(args | {"split":"locked-test"}), unlock="cfg")
    with pytest.raises(ValueError, match="smoke"):
        registry.begin(**(args | {"smoke":True,"split":"validation"}))


def test_whole_day_splits():
    from arblab.hyperliquid_copy.trial_registry import chronological_splits
    spans = chronological_splits(T,T+timedelta(days=7))
    assert spans["development"] == (T,T+timedelta(days=4))
    assert spans["validation"] == (T+timedelta(days=4),T+timedelta(days=5))
    assert spans["locked-test"][1] == T+timedelta(days=7)
    with pytest.raises(ValueError):
        chronological_splits(T,T+timedelta(days=4))


def test_terminal_trial_status_can_be_reused_without_new_run(tmp_path):
    from arblab.hyperliquid_copy.trial_registry import TrialRegistry
    registry = TrialRegistry(tmp_path/"trials.jsonl")
    trial = registry.begin(study_id="x",dataset_hash="d",config_hash="c",split="development",smoke=True)
    registry.complete(trial,dict(sharpe=None,total_return=0,max_drawdown=0),artifact_directory="existing")
    assert registry.status(trial)["artifact_directory"] == "existing"
    assert registry.status(trial)["event"] == "completed"
