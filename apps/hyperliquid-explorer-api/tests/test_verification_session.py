import importlib
import sys

import pytest

from arblab.hyperliquid_copy import download, derived_cache_resources


def test_guarded_session_restores_original_functions_after_failure(tmp_path):
    from hyperliquid_explorer_api.lab_verification_session import verification_session

    path = tmp_path / "input"
    path.write_bytes(b"data")
    original = download.file_hash
    with pytest.raises(RuntimeError, match="caller failed"):
        with verification_session("guarded-session-v1") as session:
            assert download.file_hash is session
            assert derived_cache_resources.file_hash is session
            assert session(path) == original(path)
            raise RuntimeError("caller failed")
    assert download.file_hash is original
    assert derived_cache_resources.file_hash is original


def test_full_policy_does_not_install_cache():
    from hyperliquid_explorer_api.lab_verification_session import verification_session

    original = download.file_hash
    with verification_session("full") as session:
        assert session is None
        assert download.file_hash is original


def test_unknown_policy_fails_before_hash_substitution():
    from hyperliquid_explorer_api.lab_verification_session import verification_session

    original = download.file_hash
    with pytest.raises(ValueError, match="verification policy"):
        with verification_session("unchecked"):
            pytest.fail("entered unknown policy")
    assert download.file_hash is original


def test_late_import_is_restored_at_session_end():
    from hyperliquid_explorer_api.lab_verification_session import verification_session

    original = download.file_hash
    name = "arblab.hyperliquid_copy.registration_provenance"
    sys.modules.pop(name, None)
    with verification_session("guarded-session-v1") as session:
        module = importlib.import_module(name)
        assert module.file_hash is session
    assert module.file_hash is original


def test_fingerprint_includes_runtime_verification_implementation():
    from hyperliquid_explorer_api.lab_jobs import source_fingerprint

    hashes = source_fingerprint()
    assert "api/lab_file_hash_session.py" in hashes
    assert "api/lab_verification_session.py" in hashes
    assert "api/lab_worker.py" in hashes
