"""Verify append-only job progress against preserved transition evidence."""

from pathlib import Path

from .archive_transition_snapshot import _encoded, _relative, _state, _verify_pin


def verify_progress(store, snapshot, baseline, *, recheck_content):
    """Allow new records/reservations and only proven owned-payload disposal."""
    from .archive_job_qualification import pin, verify_report
    from .prefix_qualification import _engine

    state = _state(store)
    if state["metadata_sha256"] != snapshot["metadata_sha256"]:
        raise ValueError("Transition original job changed")
    records = {(e["batch"], e["stage"]): e for e in state["records"]}
    for old in snapshot["records"]:
        if _encoded(records.get((old["batch"], old["stage"]))) != _encoded(old):
            raise ValueError("Transition original artifact changed")
    cleanup = {row[2]: row for row in state["cleanup"]}
    for old in snapshot["cleanup"]:
        if _encoded(cleanup.get(old[2])) != _encoded(old):
            raise ValueError("Transition original cleanup changed")
        if old[-1] == "deleted" and _relative(store.root, old[2]).exists():
            raise ValueError("Deleted transition payload recreated")
    current, original = state["budget"], snapshot["budget"]
    if any(current[k] != original[k] for k in ("path", "scope", "max_bytes")):
        raise ValueError("Transition original budget changed")
    if not set(map(tuple, original["reservations"])) <= set(
        map(tuple, current["reservations"])
    ):
        raise ValueError("Transition spending history lost")
    checked = set()
    for entry in snapshot["retained"]:
        path = _relative(store.root, entry["path"])
        row = cleanup.get(entry["path"])
        if row is not None:
            batch, stage, _, digest, size, qualification_sha, status = row
            if (
                stage not in ("raw", "normalized")
                or status not in ("intent", "deleted")
                or (digest, size) != (entry["sha256"], entry["bytes"])
            ):
                raise ValueError("Invalid transition disposal evidence")
            record = records.get((batch, stage))
            qualified = records.get((batch, "qualified"))
            if (
                not record
                or not qualified
                or Path(record["path"]).parent != path.parent
                or qualified["sha256"] != qualification_sha
            ):
                raise ValueError("Transition disposal lacks qualified ownership")
            if batch not in checked:
                previous = pin(records[batch - 1, "qualified"]) if batch else None
                engine = _engine()
                if batch < snapshot["boundary"]:
                    engine = snapshot["old_engine"]["qualification"]
                elif batch == snapshot["boundary"]:
                    previous = baseline
                verify_report(
                    store,
                    batch,
                    pin(qualified),
                    engine=engine,
                    previous=previous,
                    recheck_content=recheck_content,
                )
                checked.add(batch)
            if status == "deleted" and path.exists():
                raise ValueError("Deleted transition payload recreated")
            if not path.exists():
                continue  # Durable intent + qualified exact payload pin permits crash recovery.
        if recheck_content or path.name == "manifest.json":
            _verify_pin(store.root, entry)
        elif not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError("Transition retained payload changed")
    return state
