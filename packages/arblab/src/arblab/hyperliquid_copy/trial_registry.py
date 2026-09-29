"""Append-only, advisory-locked experiment state. Not a security boundary."""
from contextlib import contextmanager
from datetime import datetime, timedelta
import fcntl
import json
import os
from pathlib import Path

from .contracts import UTC, canonical_json, semantic_hash, utc


def chronological_splits(start, end):
    start, end = utc(start), utc(end)
    if any(t.hour or t.minute or t.second or t.microsecond for t in (start,end)):
        raise ValueError("whole UTC days required")
    n = (end-start).days
    if n < 5:
        raise ValueError("at least five days required")
    a, b = start+timedelta(days=int(.6*n)), start+timedelta(days=int(.6*n)+int(.2*n))
    return {"development":(start,a), "validation":(a,b), "locked-test":(b,end)}


class TrialRegistry:
    def __init__(self, path):
        self.path = Path(path)

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.with_suffix(self.path.suffix+".lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:
                rows = [json.loads(line) for line in self.path.read_text().splitlines()] if self.path.exists() else []
                yield rows
            finally:
                fcntl.flock(lock,fcntl.LOCK_UN)

    def _append(self,row):
        with self.path.open("ab") as file:
            file.write(canonical_json(row)+b"\n")
            file.flush()
            os.fsync(file.fileno())

    def status(self,trial_id):
        with self._locked() as rows:
            matches = [r for r in rows if r.get("trial_id") == trial_id]
            if not matches:
                raise ValueError("unknown trial")
            return matches[-1]

    def begin(self, *, study_id, dataset_hash, config_hash, split, smoke, unlock=None):
        if split not in ("development","validation","locked-test"):
            raise ValueError("unknown split")
        if smoke and split != "development":
            raise ValueError("smoke cannot enter research splits")
        if unlock is not None and split != "locked-test":
            raise ValueError("unlock only valid for locked-test")
        key = dict(study_id=study_id,dataset_hash=dataset_hash,config_hash=config_hash,split=split,smoke=smoke)
        trial_id = semantic_hash(key)
        with self._locked() as rows:
            study = [r for r in rows if r.get("study_id") == study_id and r.get("dataset_hash") == dataset_hash]
            selected = next((r for r in study if r["event"] == "selected"),None)
            if split != "development":
                if not selected or selected["config_hash"] != config_hash:
                    raise ValueError("configuration is not the selected winner")
                if any(r.get("split") == split and r["event"] == "completed" for r in study):
                    raise ValueError("split may complete only once")
            elif selected:
                raise ValueError("development closed after selection")
            if split == "locked-test":
                if unlock != config_hash:
                    raise ValueError("matching unlock token required")
                validation = next((r for r in study if r["event"] == "completed" and r["split"] == "validation"),None)
                if not validation or not validation["validation_passed"]:
                    raise ValueError("validation gate not passed")
            prior = [r for r in study if r.get("trial_id") == trial_id]
            if prior:
                if prior[-1]["event"] == "failed":
                    raise ValueError("failed trial needs a new study id")
                return trial_id
            self._append(key | dict(trial_id=trial_id,event="started",created_at=datetime.now(UTC)))
        return trial_id

    def complete(self,trial_id,metrics, *, artifact_directory="", failed=False):
        with self._locked() as rows:
            matches = [r for r in rows if r.get("trial_id") == trial_id]
            if not matches or matches[-1]["event"] != "started":
                raise ValueError("trial missing or already finalized")
            passed = (not failed and metrics.get("sharpe") is not None and float(metrics["sharpe"]) > 0 and
                      float(metrics.get("total_return",0)) > 0 and float(metrics.get("max_drawdown",1)) < .3)
            self._append(matches[0] | dict(event="failed" if failed else "completed",metrics=metrics,
                                          validation_passed=passed,artifact_directory=artifact_directory,
                                          created_at=datetime.now(UTC)))

    def select(self,study_id,dataset_hash):
        with self._locked() as rows:
            study = [r for r in rows if r.get("study_id") == study_id and r.get("dataset_hash") == dataset_hash]
            selected = next((r for r in study if r["event"] == "selected"),None)
            if selected:
                return selected["config_hash"]
            eligible = [r for r in study if r["event"] == "completed" and r["split"] == "development"
                        and not r["smoke"] and r["metrics"].get("sharpe") is not None]
            if not eligible:
                raise ValueError("no completed eligible development trials")
            winner = min(eligible,key=lambda r:(-float(r["metrics"]["sharpe"]),-float(r["metrics"]["total_return"]),r["config_hash"]))
            self._append(dict(event="selected",study_id=study_id,dataset_hash=dataset_hash,
                              config_hash=winner["config_hash"],created_at=datetime.now(UTC)))
            return winner["config_hash"]
