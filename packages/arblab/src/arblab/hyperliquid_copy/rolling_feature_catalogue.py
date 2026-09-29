"""Bounded metadata selection for rolling anchors and obsolete feature days."""

from contextlib import contextmanager
from datetime import datetime, timedelta
import json
from pathlib import Path

from .cache_retirement_inventory import _descriptor, _identity, _namespace, _rows
from .cache_retirement_journal import INTENT, load_journal
from .contracts import utc
from .derived_cache_policy import _PinnedFile
from .derived_cache_resources import METADATA_BYTES
from .derived_publication import _encode, _inputs
from .download import file_hash
from .feature_history import FeatureHistory
from .feature_publication import (
    FeatureDay,
    KIND as FEATURE_KIND,
    feature_engine,
    _context,
)
from .feature_resume_anchor import (
    FeatureAnchor,
    KIND as ANCHOR_KIND,
    _engine as anchor_engine,
)
from .rolling_anchor_guard import AnchorGuard
from .rolling_retirement_inventory import (
    _same_context,
    _state,
    _target_binding,
    owner_context,
)

MAX_DAYS = 732
MAX_INPUT_BYTES = 16 * 1024**2
MAX_TARGET_FILES = 10000


def _matches(inputs, expected):
    return all(inputs.get(key) == value for key, value in expected.items())


def _timestamp(value):
    if type(value) is not str or not 1 <= len(value) <= 40:
        raise ValueError("Invalid rolling anchor timestamp")
    return utc(datetime.fromisoformat(value))


def _scope(resources, pin, coins, semantics, validation, execution_policy_name=None):
    def context():
        return _encode(
            _inputs(
                "rolling_catalogue",
                dict(
                    pin=pin,
                    coins=coins,
                    semantics=semantics,
                    execution_policy=execution_policy_name,
                ),
            )
        )

    frozen = context()
    code = file_hash(Path(__file__))
    if context() != frozen:
        raise ValueError("Rolling catalogue caller context changed")
    history = FeatureHistory(resources, pin, coins, semantics)
    identity = _identity(Path(pin["path"]))
    expected = dict(
        report_sha256=pin["sha256"],
        origin=history.origin.date().isoformat(),
        coins=list(coins),
        semantics=semantics,
        execution_policy=execution_policy_name,
    )

    def check():
        if file_hash(Path(__file__)) != code:
            raise ValueError("Rolling catalogue code changed")
        history._verify()
        if validation is not None:
            validation()
        if context() != frozen:
            raise ValueError("Rolling catalogue caller context changed")
        if _identity(Path(pin["path"])) != identity:
            raise ValueError("Rolling catalogue source changed")
        resources.lease.check()

    check()
    return history, expected, check


@contextmanager
def _catalogue(resources, check):
    check()
    namespace = _namespace(resources)
    from .query_directory_pin import pin_directory

    with (
        pin_directory(resources.root, namespace[0]),
        pin_directory(resources.root / "artifacts", namespace[1]),
        _PinnedFile(resources.path, METADATA_BYTES // 2) as database,
        _PinnedFile(resources.marker, 4096) as marker,
        resources._connect() as db,
    ):
        version = db.execute("PRAGMA data_version").fetchone()
        yield db
        check()
        if (
            db.execute("PRAGMA data_version").fetchone() != version
            or _namespace(resources) != namespace
        ):
            raise ValueError("Rolling catalogue changed during selection")
        database.check()
        marker.check()


def _documents(resources, db):
    for row in _rows(db):
        yield _descriptor(resources, db, *row), json.loads(row[1])


def select_anchor(
    resources,
    pin,
    coins,
    semantics,
    start,
    end,
    *,
    validation=None,
    execution_policy_name=None,
):
    history, expected, check = _scope(
        resources, pin, coins, semantics, validation, execution_policy_name
    )
    start, end = utc(start), utc(end)
    if not history.origin <= start < end <= history.finish or end - start > timedelta(
        days=732
    ):
        raise ValueError("Invalid rolling catalogue interval")
    first = start.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0)
    if last < end:
        last += timedelta(days=1)
    selected, priority = None, None
    engine = anchor_engine()
    with _catalogue(resources, check) as db:
        for publication, data in _documents(resources, db):
            inputs = data["inputs"]
            if (
                data["kind"] != ANCHOR_KIND
                or not _matches(inputs, expected)
                or inputs.get("engine") != engine
            ):
                continue
            lower, upper = (
                _timestamp(inputs.get("first")),
                _timestamp(inputs.get("cutoff")),
            )
            if (
                inputs["first"] != lower.isoformat()
                or inputs["cutoff"] != upper.isoformat()
                or lower != lower.replace(hour=0, minute=0, second=0, microsecond=0)
                or upper != upper.replace(hour=0, minute=0, second=0, microsecond=0)
                or not history.origin <= lower < upper <= history.finish
            ):
                raise ValueError("Invalid rolling anchor metadata interval")
            current = (lower, upper, publication.key)
            if (
                lower <= first
                and upper <= last
                and (priority is None or current > priority)
            ):
                selected, priority = inputs, current
        result = None if selected is None else FeatureAnchor(resources, pin, selected)
        stats = None if result is None else result._stats()
        if result is not None:
            result.verify()
    if result is not None and result._stats() != stats:
        raise ValueError("Selected rolling anchor identity changed")
    return result


def require_no_pending(
    resources,
    pin,
    coins,
    semantics,
    *,
    validation=None,
    execution_policy_name=None,
):
    """Before a cold start, reject matching unfinished work without adopting it."""
    _, expected, check = _scope(
        resources, pin, coins, semantics, validation, execution_policy_name
    )
    with _catalogue(resources, check) as db:
        for _, data in _documents(resources, db):
            receipt = data["inputs"]
            if data["kind"] != INTENT or "owner" not in receipt:
                continue
            if not _matches(owner_context(resources, db, receipt["owner"]), expected):
                continue
            body, publication = load_journal(resources, receipt)
            if _state(resources, db, receipt, body, publication) != "complete":
                raise ValueError(
                    "Unresolved rolling retirement owner blocks cold start"
                )


def expired_feature_days(
    resources, pin, anchor, *, validation=None, execution_policy_name=None
):
    _, _, check = _scope(
        resources,
        pin,
        anchor.inputs["coins"],
        anchor.inputs["semantics"],
        validation,
        execution_policy_name,
    )
    protection = AnchorGuard(resources, pin, anchor, check)
    days, seen, encoded_bytes, snapshots = [], set(), 0, []
    engine = feature_engine()
    with _catalogue(resources, protection.check) as db:
        for publication, data in _documents(resources, db):
            if (
                data["kind"] != FEATURE_KIND
                or not _same_context(data, anchor)
                or data["inputs"].get("engine") != engine
            ):
                continue
            inputs, day, cutoff = _context(data["inputs"])
            if cutoff > anchor.first:
                continue
            if day in seen:
                raise ValueError("Ambiguous obsolete feature day variants")
            seen.add(day)
            encoded_bytes += len(_encode(inputs))
            if len(days) >= MAX_DAYS or encoded_bytes > MAX_INPUT_BYTES:
                raise ValueError("Rolling feature target resource limit exceeded")
            if len(snapshots) + len(publication.artifacts) > MAX_TARGET_FILES:
                raise ValueError("Rolling feature target resource limit exceeded")
            snapshots.extend(
                (resources.root / pin.path, _identity(resources.root / pin.path))
                for pin in publication.artifacts
            )
            _target_binding(data, anchor, pin)
            days.append(FeatureDay(resources, inputs))
            protection.check()
    if any(_identity(path) != identity for path, identity in snapshots):
        raise ValueError("Obsolete feature target identity changed")
    return tuple(sorted(days, key=lambda day: day.day))
