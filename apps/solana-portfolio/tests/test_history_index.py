import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))


class Ledger:
    def __init__(self, count):
        self.signatures = [self.signature(i) for i in range(count, 0, -1)]
        self.calls = []
        self.missing = set()
        self.fail_before = None

    @staticmethod
    def signature(i):
        return {"signature": f"sig-{i}", "blockTime": i, "slot": i, "err": None}

    def rpc(self, method, params):
        self.calls.append((method, params))
        if method == "getSignaturesForAddress":
            options = params[1]
            assert options["limit"] == 1000
            cursor = options.get("before")
            if cursor == self.fail_before and cursor is not None:
                raise RuntimeError("https://rpc/?api-key=SECRET")
            start = next(
                (
                    i + 1
                    for i, item in enumerate(self.signatures)
                    if item["signature"] == cursor
                ),
                0,
            )
            return self.signatures[start : start + 1000]
        signature = params[0]
        if signature in self.missing:
            return None
        return {"signature": signature, "meta": {"err": None}}


def parse(tx, wallet, signature):
    return {
        "id": signature,
        "time": int(signature.split("-")[1]),
        "type": "unclassified",
    }


def run(tmp_path, ledger, **kwargs):
    from api.history_index import sync_history

    return sync_history(
        "11111111111111111111111111111111",
        directory=tmp_path,
        rpc_call=ledger.rpc,
        parser=parse,
        **kwargs,
    )


def test_discovers_all_signature_pages_and_decodes_more_than_40(tmp_path):
    ledger = Ledger(2015)
    result = run(tmp_path, ledger)
    assert len(result["records"]) == 2015
    assert result["status"]["discovered"] == 2015
    assert result["status"]["processed"] == 2015
    assert result["status"]["oldest"] == 1
    assert result["status"]["discoveryComplete"] is True
    assert result["status"]["decodingComplete"] is True
    assert result["status"]["complete"] is True
    assert result["records"][0]["time"] == 2015


def test_missing_transactions_are_retried_without_refetching_cached_ones(tmp_path):
    ledger = Ledger(5)
    ledger.missing = {"sig-3"}
    result = run(tmp_path, ledger)
    assert result["status"]["missing"] == 1
    assert result["status"]["complete"] is False
    assert result["status"]["discoveryComplete"] is True
    ledger.missing.clear()
    ledger.calls.clear()
    result = run(tmp_path, ledger)
    assert result["status"]["complete"] is True
    assert [
        params[0] for method, params in ledger.calls if method == "getTransaction"
    ] == ["sig-3"]


def test_restart_reuses_cache_and_resumes_undecoded_transactions(tmp_path):
    ledger = Ledger(12)
    first = run(tmp_path, ledger, max_transactions=4)
    assert first["status"]["discoveryComplete"] is True
    assert first["status"]["processed"] == 4
    ledger.calls.clear()
    result = run(tmp_path, ledger)
    assert result["status"]["processed"] == 12
    assert len([1 for method, _ in ledger.calls if method == "getTransaction"]) == 8


def test_refresh_only_decodes_new_head_signatures(tmp_path):
    ledger = Ledger(7)
    run(tmp_path, ledger)
    ledger.signatures.insert(0, ledger.signature(8))
    ledger.calls.clear()
    result = run(tmp_path, ledger)
    assert len(result["records"]) == 8
    assert [
        params[0] for method, params in ledger.calls if method == "getTransaction"
    ] == ["sig-8"]


def test_failed_page_preserves_cursor_and_sanitizes_errors(tmp_path):
    ledger = Ledger(1500)
    ledger.fail_before = "sig-501"
    partial = run(tmp_path, ledger, max_transactions=0)
    assert partial["status"]["discovered"] == 1000
    assert partial["status"]["discoveryComplete"] is False
    assert "SECRET" not in partial["status"]["error"]
    ledger.fail_before = None
    result = run(tmp_path, ledger, max_transactions=0)
    assert result["status"]["discovered"] == 1500
    assert result["status"]["discoveryComplete"] is True
    assert result["status"]["error"] is None


def test_wallet_caches_are_isolated_and_seed_requires_explicit_wallet(tmp_path):
    from api.history_index import seed_signatures, sync_history

    ledger = Ledger(3)
    wallet_a = "11111111111111111111111111111111"
    wallet_b = "So11111111111111111111111111111111111111112"
    seed_signatures(wallet_a, ledger.signatures, directory=tmp_path)

    def empty_rpc(method, params):
        assert method == "getSignaturesForAddress"
        return []

    result = sync_history(
        wallet_b, directory=tmp_path, rpc_call=empty_rpc, parser=parse
    )
    assert result["status"]["discovered"] == 0
    assert result["records"] == []


def test_failed_signatures_are_accounted_for_without_successful_activity(tmp_path):
    ledger = Ledger(3)
    ledger.signatures[0]["err"] = {"InstructionError": [0, "Failed"]}
    result = run(tmp_path, ledger)
    assert len(result["records"]) == 2
    assert result["status"]["failed"] == 1
    assert result["status"]["processed"] == 3
    assert result["status"]["complete"] is True


def test_invalid_wallet_cannot_escape_cache_directory(tmp_path):
    from api.history_index import sync_history

    with pytest.raises(ValueError):
        sync_history(
            "../../other", directory=tmp_path, rpc_call=Ledger(0).rpc, parser=parse
        )


def test_interrupted_multi_page_head_refresh_does_not_skip_uncached_gap(tmp_path):
    ledger = Ledger(2)
    run(tmp_path, ledger)
    ledger.signatures = [ledger.signature(i) for i in range(1505, 0, -1)]
    ledger.fail_before = "sig-506"
    partial = run(tmp_path, ledger, max_transactions=0)
    assert partial["status"]["discovered"] == 1002
    assert partial["status"]["discoveryComplete"] is False
    ledger.fail_before = None
    result = run(tmp_path, ledger, max_transactions=0)
    assert result["status"]["discovered"] == 1505
    assert result["status"]["discoveryComplete"] is True


def test_reads_are_nonblocking_and_only_start_one_wallet_worker(tmp_path):
    import threading
    import time
    from api.history_index import read_history

    entered, release = threading.Event(), threading.Event()
    calls = []

    def blocking_rpc(method, params):
        calls.append(method)
        entered.set()
        assert release.wait(5)
        return []

    wallet = "11111111111111111111111111111111"
    try:
        first = read_history(
            wallet, directory=tmp_path, rpc_call=blocking_rpc, parser=parse
        )
        assert first["status"]["running"] is True
        assert entered.wait(2)
        second = read_history(
            wallet, directory=tmp_path, rpc_call=blocking_rpc, parser=parse
        )
        assert second["status"]["running"] is True
        assert calls == ["getSignaturesForAddress"]
    finally:
        release.set()
    for _ in range(100):
        result = read_history(
            wallet, directory=tmp_path, rpc_call=blocking_rpc, parser=parse
        )
        if not result["status"]["running"]:
            break
        time.sleep(0.01)
    assert result["status"]["complete"] is True


def test_parser_upgrade_reclassifies_cached_raw_without_network_download(
    tmp_path, monkeypatch
):
    from api import history_index

    ledger = Ledger(3)
    run(tmp_path, ledger)
    ledger.calls.clear()
    monkeypatch.setattr(history_index, "PARSER_VERSION", 99)
    result = history_index.sync_history(
        "11111111111111111111111111111111",
        directory=tmp_path,
        rpc_call=ledger.rpc,
        parser=lambda tx, wallet, signature: {
            "id": signature,
            "time": 1,
            "type": "buy",
            "protocol": "Raydium",
        },
    )
    assert all(record["protocol"] == "Raydium" for record in result["records"])
    assert not any(method == "getTransaction" for method, _ in ledger.calls)
