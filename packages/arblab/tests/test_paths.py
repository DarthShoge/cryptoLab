"""Tests for repository resource path resolution."""

from pathlib import Path

from arblab.paths import (
    fixture_path,
    notebook_price_cache_dir,
    price_cache_dir,
    repo_root,
    reports_dir,
)


def test_paths_are_absolute_and_independent_of_cwd(monkeypatch, tmp_path):
    expected_root = Path(__file__).resolve().parents[3]

    monkeypatch.delenv("CRYPTOLAB_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)

    assert repo_root() == expected_root
    assert fixture_path("prices/sample.csv") == expected_root / "data/fixtures/prices/sample.csv"
    assert reports_dir() == expected_root / "reports"
    assert price_cache_dir() == expected_root / ".price_cache"
    assert notebook_price_cache_dir() == expected_root / "notebooks/.price_cache"
    assert all(
        path.is_absolute()
        for path in (
            repo_root(),
            fixture_path("sample.csv"),
            reports_dir(),
            price_cache_dir(),
            notebook_price_cache_dir(),
        )
    )


def test_environment_override_is_resolved_against_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CRYPTOLAB_ROOT", "alternate-root")

    expected_root = (tmp_path / "alternate-root").resolve()
    assert repo_root() == expected_root
    assert fixture_path("sample.csv") == expected_root / "data/fixtures/sample.csv"
    assert reports_dir() == expected_root / "reports"
    assert price_cache_dir() == expected_root / ".price_cache"
    assert notebook_price_cache_dir() == expected_root / "notebooks/.price_cache"


def test_path_helpers_do_not_create_directories(monkeypatch, tmp_path):
    root = tmp_path / "missing-root"
    monkeypatch.setenv("CRYPTOLAB_ROOT", str(root))

    fixture_path("nested/sample.csv")
    reports_dir()
    price_cache_dir()
    notebook_price_cache_dir()

    assert not root.exists()
