"""Bind canonical validation to frozen job artifacts, without disposal authority."""

from pathlib import Path
from tempfile import TemporaryDirectory

from .archive_job_artifacts import read
from .download import file_hash
from .prefix_qualification import _engine, _previous, _verify_all, qualify_prefix


def pin(record):
    return dict(path=str(record["path"]), sha256=record["sha256"])


def verify(store, index, report_pin, *, recheck_content=True):
    from .archive_engine_transition import active_transition

    transition = active_transition(store, recheck_content=recheck_content)
    previous = pin(store.records()[index - 1, "qualified"]) if index else None
    engine = _engine()
    if transition:
        if index < transition["boundary"]:
            engine = transition["old_engine"]["qualification"]
        elif index == transition["boundary"]:
            previous = transition["baseline"]
    return verify_report(
        store,
        index,
        report_pin,
        engine=engine,
        previous=previous,
        recheck_content=recheck_content,
    )


def verify_report(store, index, report_pin, *, engine, previous, recheck_content=True):
    """Exact report membership check; caller supplies an authorized generation."""
    report = _previous(report_pin, engine)
    records = store.records()
    manifests = [pin(records[n, "compact"]) for n in range(index + 1)]
    batches = store.metadata["plan"]["batches"]
    if (
        report["manifests"] != manifests
        or report["previous"] != previous
        or report["coins"] != store.metadata["coins"]
        or report["source_start"] != batches[0]["start"]
        or report["source_end"] != batches[index]["end"]
        or report["research_eligible"] is not False
        or report["raw_disposal_authorized"] is not False
    ):
        raise ValueError("Qualification report does not match frozen job prefix")
    raw_pins = [pin(records[n, "raw"]) for n in range(index + 1)]
    if [pin(s) for s in report["raw_sources"]] != raw_pins:
        raise ValueError("Qualification raw provenance does not match job")
    expected = []
    for manifest in manifests:
        path = Path(manifest["path"])
        if file_hash(path) != manifest["sha256"]:
            raise ValueError("Qualification compact manifest identity changed")
        for entry in read(path)["files"]:
            expected.append(
                dict(
                    path=str(path.parent / entry["name"]),
                    sha256=entry["sha256"],
                    bytes=entry["bytes"],
                    rows=entry["rows"],
                )
            )
    if [
        {k: e[k] for k in ("path", "sha256", "bytes", "rows")} for e in report["files"]
    ] != expected:
        raise ValueError("Qualification canonical file membership changed")
    if recheck_content:
        _verify_all(manifests, report["files"], report["raw_sources"], report_pin)
    return report_pin


def complete(store, index, progress=None):
    from .archive_engine_transition import active_transition

    records = store.records()
    path = store.find(index, "qualified")
    previous = pin(records[index - 1, "qualified"]) if index else None
    transition = active_transition(store)
    if transition and index == transition["boundary"]:
        previous = transition["baseline"]
    manifests = [records[n, "compact"]["path"] for n in range(index + 1)]
    if path is None:
        result = qualify_prefix(
            manifests,
            previous=previous,
            output_root=store.stage_root(index, "qualified"),
            temp_root=store.root,
            progress=progress,
        )
        path = Path(result["path"])
    elif (index, "qualified") not in records:
        # Publication survived but SQLite has no trusted pin. Recompute all
        # semantics, including pruning bounds, rather than trusting a new hash
        # of whatever bytes now occupy the orphan report path.
        with TemporaryDirectory(
            prefix="qualification_recovery_", dir=store.root
        ) as scratch:
            repeated = qualify_prefix(
                manifests,
                previous=previous,
                output_root=scratch,
                temp_root=scratch,
                progress=progress,
            )
            if repeated["sha256"] != file_hash(path):
                raise ValueError(
                    "Uncommitted qualification report differs from revalidation"
                )
    result = dict(path=str(path), sha256=file_hash(path))
    verify(store, index, result)
    return path
