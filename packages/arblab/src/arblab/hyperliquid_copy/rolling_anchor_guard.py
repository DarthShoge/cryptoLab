"""Bounded metadata guard after full anchor verification, across cache writes."""

from pathlib import Path

from .cache_retirement_inventory import _descriptor
from .derived_publication import MAX_DESCRIPTOR_BYTES
from .prefix_qualification import _previous, _engine as qualification_engine
from .qualified_source_session import _identity

MAX_FILES = 15002
MAX_DESCRIPTOR_TOTAL = 16 * 1024**2


class AnchorGuard:
    def __init__(self, resources, report_pin, anchor, context_check):
        self.resources, self.context_check = resources, context_check
        context_check()
        report = _previous(report_pin, qualification_engine())
        paths = {Path(report_pin["path"])}
        canonical = set()
        for entry in report["files"]:
            path = Path(entry["path"])
            paths.add(path)
            canonical.add(path)
        publications = (anchor.publication, *(day.publication for day in anchor.days))
        self.records = []
        total = 0
        with resources._connect() as db:
            for publication in publications:
                row = self._row(db, publication.key)
                if row is None or _descriptor(resources, db, *row) != publication:
                    raise ValueError("Protected anchor publication changed")
                total += len(row[1].encode())
                if total > MAX_DESCRIPTOR_TOTAL:
                    raise ValueError("Anchor guard descriptor limit exceeded")
                self.records.append(row)
                for pin in publication.artifacts:
                    paths.add(resources.root / pin.path)
                    if len(paths) > MAX_FILES:
                        raise ValueError("Anchor guard file limit exceeded")
        self.records = tuple(self.records)
        self.files = tuple(
            (path, path in canonical, _identity(path, canonical=path in canonical))
            for path in sorted(paths)
        )
        anchor.verify()
        self.check()

    @staticmethod
    def _row(db, key):
        return db.execute(
            "SELECT key, CASE WHEN length(CAST(descriptor AS BLOB)) BETWEEN 1 AND ? "
            "THEN descriptor END, sha256 FROM publications WHERE key=?",
            (MAX_DESCRIPTOR_BYTES, key),
        ).fetchone()

    def check(self):
        self.context_check()
        # The retirement transaction changes only its own catalogue rows. These
        # protected publications must stay identical, including allocation pins.
        with self.resources._connect() as db:
            for expected in self.records:
                row = self._row(db, expected[0])
                if row != expected:
                    raise ValueError("Protected anchor publication changed")
                _descriptor(self.resources, db, *row)
        self.context_check()
        for path, canonical, identity in self.files:
            if _identity(path, canonical=canonical) != identity:
                raise ValueError("Protected anchor/source identity changed")
