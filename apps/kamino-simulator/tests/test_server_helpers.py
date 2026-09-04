"""Tests for the functional test server lifecycle helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from kamino_simulator import server_helpers


def test_ephemeral_port_is_allocated_by_binding_localhost_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound_addresses: list[tuple[str, int]] = []

    class FakeSocket:
        def __enter__(self) -> "FakeSocket":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            bound_addresses.append(address)

        def getsockname(self) -> tuple[str, int]:
            return ("127.0.0.1", 43123)

    monkeypatch.setattr(server_helpers.socket, "socket", FakeSocket)

    port = server_helpers.ephemeral_local_port()

    assert bound_addresses == [("127.0.0.1", 0)]
    assert port == 43123


def test_readiness_reports_early_process_exit_with_log(tmp_path: Path) -> None:
    log_path = tmp_path / "streamlit.log"
    log_path.write_text("startup exploded", encoding="utf-8")

    class ExitedProcess:
        def poll(self) -> int:
            return 7

    with pytest.raises(RuntimeError, match="startup exploded"):
        server_helpers.wait_for_server(ExitedProcess(), "http://127.0.0.1:1", log_path)


def test_termination_escalates_to_kill_after_timeout() -> None:
    events: list[object] = []

    class StubbornProcess:
        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout: float) -> None:
            events.append(("wait", timeout))
            if events.count("kill") == 0:
                raise subprocess.TimeoutExpired("streamlit", timeout)

        def kill(self) -> None:
            events.append("kill")

    server_helpers.stop_process(StubbornProcess(), timeout=2)

    assert events == ["terminate", ("wait", 2), "kill", ("wait", 2)]
