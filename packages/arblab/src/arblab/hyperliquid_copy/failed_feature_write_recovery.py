"""Explicit quarantine of exact, unpublished allocations from a failed write."""

import json
from pathlib import Path
import re

from .derived_cache_resources import _sync
from .download import file_hash
from .failed_ranking_staging_recovery import _tree, _verify_tree


def quarantine(resources, tokens, destination):
    resources.lease.check()
    if (
        not 1 <= len(tokens) <= 32
        or len(set(tokens)) != len(tokens)
        or any(
            not isinstance(token, str) or not re.fullmatch("[a-f0-9]{32}", token)
            for token in tokens
        )
    ):
        raise ValueError("Expected exact bounded recovery tokens")
    destination = Path(destination).absolute()
    if (
        destination.exists()
        or destination.is_symlink()
        or destination.is_relative_to(resources.root)
    ):
        raise ValueError("Expected new quarantine outside the cache")
    if not destination.parent.is_dir() or any(
        p.is_symlink() for p in destination.parents
    ):
        raise ValueError("Unsafe quarantine parent")
    moved, committed = [], False
    try:
        with resources._connect() as db, db:
            db.execute("BEGIN IMMEDIATE")
            resources._audit(db)
            rows = [
                db.execute(
                    "SELECT * FROM allocations WHERE token=?", (token,)
                ).fetchone()
                for token in tokens
            ]
            if any(row is None for row in rows):
                raise ValueError("Missing recovery allocation")
            for (raw,) in db.execute("SELECT descriptor FROM publications"):
                if any(token in raw for token in tokens):
                    raise ValueError(
                        "Recovery allocation is referenced by a publication"
                    )
            trees, inventory = [], []
            for row in rows:
                path = resources._path(row[1])
                tree = _tree(path, row[2])
                if not tree:
                    raise ValueError("Missing recovery output")
                trees.append(tree)
                for item, *_ in tree:
                    if item.is_file():
                        inventory.append(
                            dict(
                                path=item.relative_to(resources.root).as_posix(),
                                bytes=item.stat().st_size,
                                sha256=file_hash(item),
                            )
                        )
            for tree in trees:
                _verify_tree(tree)
            destination.mkdir()
            plan = dict(
                schema=1, cache=str(resources.root), allocations=rows, files=inventory
            )
            plan_path = destination / "plan.json"
            with plan_path.open("x") as handle:
                json.dump(plan, handle, indent=2)
            _sync(plan_path)
            _sync(destination)
            for row, tree in zip(rows, trees):
                resources.lease.check()
                _verify_tree(tree)
                source = resources._path(row[1])
                target = destination / source.name
                if target.exists():
                    raise ValueError("Quarantine filename collision")
                source.rename(target)
                moved.append((source, target))
                _sync(source.parent)
                _sync(destination)
                db.execute("DELETE FROM allocations WHERE token=?", (row[0],))
            resources._audit(db)
        committed = True
    except BaseException:
        if not committed:
            for source, target in reversed(moved):
                target.rename(source)
                _sync(source.parent)
            if moved:
                _sync(destination)
        raise
    result = dict(
        allocations=len(tokens),
        files=len(inventory),
        bytes=sum(f["bytes"] for f in inventory),
        quarantine=str(destination),
    )
    complete = destination / "complete.json"
    with complete.open("x") as handle:
        json.dump(result, handle, indent=2)
    _sync(complete)
    _sync(destination)
    return result
