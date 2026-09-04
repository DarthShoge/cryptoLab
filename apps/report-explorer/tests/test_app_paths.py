"""Report explorer application path and startup tests."""

from pathlib import Path

from arblab.paths import notebook_price_cache_dir, repo_root


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


def test_report_explorer_app_uses_repository_paths(monkeypatch, tmp_path):
    from report_explorer import app

    expected_root = Path(__file__).resolve().parents[3]
    monkeypatch.delenv("CRYPTOLAB_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    assert Path(app.__file__).resolve() == (
        expected_root / "apps" / "report-explorer" / "src" / "report_explorer" / "app.py"
    )
    assert not (expected_root / "strategy_report_app.py").exists()
    assert app.REPORT_ROOT == expected_root / "reports"

    observed = {}

    def capture_cache_dir(cache_dir, symbols):
        observed["args"] = (cache_dir, symbols)
        return {}

    monkeypatch.setattr(app, "load_price_cache", capture_cache_dir)
    assert app._load_prices.__wrapped__() == {}
    assert observed["args"] == (
        expected_root / "notebooks" / ".price_cache",
        ["SOL", "ETH"],
    )
