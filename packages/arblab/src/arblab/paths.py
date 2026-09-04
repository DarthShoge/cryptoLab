"""Compose trusted repository-relative resource names into stable paths."""

from __future__ import annotations

import os
from pathlib import Path


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def repo_root() -> Path:
    """Return the root; relative ``CRYPTOLAB_ROOT`` values resolve against cwd."""
    configured_root = os.environ.get("CRYPTOLAB_ROOT")
    if configured_root is not None:
        return Path(configured_root).resolve()
    return _default_repo_root()


def fixture_path(name: str) -> Path:
    """Compose a trusted repo-relative fixture name into an absolute path."""
    return (repo_root() / "data" / "fixtures" / name).resolve()


def reports_dir() -> Path:
    """Return the absolute directory for repository reports."""
    return (repo_root() / "reports").resolve()


def price_cache_dir() -> Path:
    """Return the absolute directory for the repository price cache."""
    return (repo_root() / ".price_cache").resolve()


def hyperliquid_cache_dir() -> Path:
    """Return the local Hyperliquid research cache without creating it."""
    return (repo_root() / ".hyperliquid_cache").resolve()


def notebook_price_cache_dir() -> Path:
    """Return the absolute directory for the historical notebook price cache."""
    return (repo_root() / "notebooks" / ".price_cache").resolve()
