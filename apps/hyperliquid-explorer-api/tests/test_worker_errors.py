import sys

import pytest

from hyperliquid_explorer_api import lab_worker


def test_worker_preserves_failure_traceback_in_local_stderr(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "argv", ["lab_worker", "--root", "/tmp/test-lab", "--id", "a" * 32]
    )

    def fail(*args):
        raise ValueError("Feature reservation exceeded")

    monkeypatch.setattr(lab_worker, "execute", fail)
    with pytest.raises(SystemExit) as exc:
        lab_worker.main()
    assert exc.value.code == 1
    assert "ValueError: Feature reservation exceeded" in capsys.readouterr().err
