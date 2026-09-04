"""Stable paths to repository resources."""

from __future__ import annotations

import os
from pathlib import Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def repo_root() -> Path:
    """Return the configured repository root as an absolute path."""
    configured_root = os.environ.get("CRYPTOLAB_ROOT")
    if configured_root is not None:
        return Path(configured_root).resolve()
    return _default_repo_root()


def fixture_path(name: str) -> Path:
    return (repo_root() / "data" / "fixtures" / name).resolve()


def reports_dir() -> Path:
    return (repo_root() / "reports").resolve()


def price_cache_dir() -> Path:
    return (repo_root() / ".price_cache").resolve()
