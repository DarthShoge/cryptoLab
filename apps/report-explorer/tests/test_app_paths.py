"""Report explorer application path and startup tests."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

from arblab.paths import notebook_price_cache_dir


def test_notebook_price_cache_dir_is_stable_and_has_no_side_effects(monkeypatch, tmp_path):
    expected_root = Path(__file__).resolve().parents[3]
    expected_cache = expected_root / "notebooks" / ".price_cache"

    monkeypatch.delenv("CRYPTOLAB_ROOT", raising=False)
    before = set(tmp_path.iterdir())
    monkeypatch.chdir(tmp_path)

    assert notebook_price_cache_dir() == expected_cache
    assert notebook_price_cache_dir().is_absolute()
    assert set(tmp_path.iterdir()) == before


def test_notebook_price_cache_dir_honors_repo_root_override(monkeypatch, tmp_path):
    configured_root = tmp_path / "alternate-root"
    monkeypatch.setenv("CRYPTOLAB_ROOT", str(configured_root))

    assert notebook_price_cache_dir() == configured_root / "notebooks" / ".price_cache"
    assert not configured_root.exists()


def test_report_explorer_app_uses_repository_paths_from_an_arbitrary_cwd(tmp_path):
    expected_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["CRYPTOLAB_ROOT"] = str(expected_root)
    script = textwrap.dedent(
        """
        import os
        from pathlib import Path
        import socket

        def block_network(*args, **kwargs):
            raise AssertionError("report explorer startup attempted a socket connection")

        socket.create_connection = block_network
        socket.socket.connect = block_network

        from report_explorer import app

        root = Path(os.environ["CRYPTOLAB_ROOT"]).resolve()
        assert Path(app.__file__).resolve() == (
            root / "apps" / "report-explorer" / "src" / "report_explorer" / "app.py"
        )
        assert app.REPORT_ROOT == root / "reports"

        observed = {}

        def capture_cache_dir(cache_dir, symbols):
            observed["args"] = (cache_dir, symbols)
            return {}

        app.load_price_cache = capture_cache_dir
        assert app._load_prices.__wrapped__() == {}
        assert observed["args"] == (
            root / "notebooks" / ".price_cache",
            ["SOL", "ETH"],
        )
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
