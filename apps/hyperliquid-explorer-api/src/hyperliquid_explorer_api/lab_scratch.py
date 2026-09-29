"""Job-owned disposable scratch, separate from datasets and published evidence."""

from pathlib import Path
import re
import shutil


def scratch_path(root, identifier):
    if not re.fullmatch(r"[a-f0-9]{32}", identifier):
        raise ValueError("Invalid scratch owner")
    root = Path(root).resolve()
    parent = root / "scratch"
    path = parent / identifier
    if (
        parent.is_symlink()
        or path.is_symlink()
        or not path.resolve().is_relative_to(root)
    ):
        raise ValueError("Unsafe scratch path")
    return path


def clean_scratch(root, identifier):
    path = scratch_path(root, identifier)
    if path.exists():
        shutil.rmtree(path)
