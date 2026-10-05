import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from api import server


def test_explicit_refresh_bypasses_live_cache(monkeypatch):
    snapshots = iter([{"asOf": 1}, {"asOf": 2}])
    monkeypatch.setattr(server, "_cache", {})
    monkeypatch.setattr(server, "live_portfolio", lambda wallet: next(snapshots))
    assert server.cached_live("wallet")["asOf"] == 1
    assert server.cached_live("wallet")["asOf"] == 1
    assert server.cached_live("wallet", refresh=True)["asOf"] == 2
