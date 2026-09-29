"""Fully verify a canonical corpus once, then guard its unchanged identity.

Sessions are process-local and immutable. A fresh session repeats all checksums;
no serialized receipt can substitute for that initial verification.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import stat

from .archive_cache import _safe
from .contracts import semantic_hash
from .derived_publication import _encode, _inputs
from .download import file_hash
from .prefix_qualification import _previous, _engine as qualification_engine
from .qualified_day import QualifiedDay, QualifiedFile, _day


def _engine():
    root = Path(__file__).parent
    return semantic_hash(
        dict(
            qualification=qualification_engine(),
            code={
                name: file_hash(root / name)
                for name in (
                    "qualified_source_session.py",
                    "qualified_day.py",
                    "archive_cache.py",
                    "contracts.py",
                    "derived_publication.py",
                    "download.py",
                )
            },
        )
    )


def _identity(path, *, canonical=False):
    try:
        _safe(path)
        info = path.lstat()
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_nlink < 1
            or not canonical
            and info.st_nlink != 1
        ):
            raise ValueError("Unsafe qualified source session file")
        return (
            info.st_dev,
            info.st_ino,
            info.st_mode,
            info.st_nlink,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
    except OSError as exc:
        raise ValueError("Missing qualified source session file") from exc


def _pin(value):
    result = _inputs("qualified_source_session", value)
    if set(result) != {"path", "sha256"} or any(
        type(result[k]) is not str for k in result
    ):
        raise ValueError("Exact pinned qualified source required")
    return result


@dataclass(frozen=True, init=False)
class QualifiedSourceSession:
    origin: object
    finish: object
    coins: tuple
    _caller: object
    _pin_bytes: bytes
    _encoded: bytes
    _files: tuple
    _identities: tuple
    _report: Path
    _report_identity: tuple
    _engine_digest: str

    def __init__(self, report_pin):
        pin = _pin(report_pin)
        frozen, engine = _encode(pin), _engine()
        report_path = Path(pin["path"]).absolute()
        report_identity = _identity(report_path)
        report = _previous(pin, qualification_engine())
        # Reuse the existing complete corpus bounds, schema and market checks.
        anchor = QualifiedDay(pin, report["source_start"])
        files = tuple(QualifiedFile.parse(row) for row in report["files"])
        # Registration retains canonical bytes via hardlinks. Pin their initial
        # topology as well as checksums; owned cache/report files remain single-link.
        identities = tuple(_identity(f.path, canonical=True) for f in files)
        for entry in files:
            entry.verify()
        inputs = dict(
            schema=1,
            pin=pin,
            origin=report["source_start"],
            finish=report["source_end"],
            coins=list(anchor.coins),
            files=len(files),
            membership_sha256=hashlib.sha256(_encode(report["files"])).hexdigest(),
            engine=engine,
        )
        for name, value in dict(
            origin=_day(report["source_start"]),
            finish=_day(report["source_end"]),
            coins=anchor.coins,
            _caller=report_pin,
            _pin_bytes=frozen,
            _encoded=_encode(inputs),
            _files=files,
            _identities=identities,
            _report=report_path,
            _report_identity=report_identity,
            _engine_digest=engine,
        ).items():
            object.__setattr__(self, name, value)
        self.verify()

    def inputs(self):
        return json.loads(self._encoded)

    def _check_identities(self):
        if (
            _identity(self._report) != self._report_identity
            or tuple(_identity(f.path, canonical=True) for f in self._files)
            != self._identities
        ):
            raise ValueError("Qualified source session file identity changed")

    def verify(self):
        self._check_identities()
        pin = json.loads(self._pin_bytes)
        if file_hash(self._report) != pin["sha256"] or _engine() != self._engine_digest:
            raise ValueError("Qualified source session report/engine changed")
        # Recheck after report/code hashing; those checks may be nontrivial.
        self.verify_identity()

    def verify_identity(self):
        """Final identity/caller guard after a consumer's expensive checks.

        Use alongside verify(), not instead of its report/engine validation.
        """
        self._check_identities()
        if _encode(_pin(self._caller)) != self._pin_bytes:
            raise ValueError("Qualified source session caller context changed")
