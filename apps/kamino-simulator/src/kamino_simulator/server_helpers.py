"""Process helpers for the Streamlit functional-test server."""

from __future__ import annotations

import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


def ephemeral_local_port() -> int:
    """Ask the OS for an available TCP port on localhost."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _diagnostics(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return f"unable to read Streamlit log: {exc}"


def wait_for_server(
    process: Any,
    url: str,
    log_path: Path,
    *,
    timeout: float = 15,
    poll_interval: float = 0.1,
) -> None:
    """Wait until Streamlit responds, failing promptly with startup diagnostics."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Streamlit exited with code {return_code}:\n{_diagnostics(log_path)}"
            )
        try:
            with urlopen(url, timeout=min(1.0, poll_interval + 0.5)) as response:
                if response.status < 500:
                    return
        except (OSError, URLError):
            time.sleep(poll_interval)

    raise RuntimeError(f"Streamlit did not become ready:\n{_diagnostics(log_path)}")


def stop_process(process: Any, *, timeout: float = 5) -> None:
    """Stop a child process, escalating to kill if graceful shutdown stalls."""
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)
