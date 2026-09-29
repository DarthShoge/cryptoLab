"""Read-only, bounded classification of anchor-owned feature retirement."""

from dataclasses import dataclass
import json

from .cache_retirement import _detached, _files
from .cache_retirement_inventory import _descriptor, _digest, _namespace, _rows
from .cache_retirement_journal import COMMITTED, INTENT, load_journal
from .derived_cache_policy import _PinnedFile
from .derived_cache_resources import METADATA_BYTES
from .derived_publication import (
    MAX_DESCRIPTOR_BYTES,
    PublishedArtifacts,
    _encode,
    publication_key,
)
from .feature_publication import KIND as FEATURE_KIND, _context as feature_context
from .feature_resume_anchor import KIND as ANCHOR_KIND
from .qualified_window import QualifiedWindow
from .query_directory_pin import pin_directory

MAX_OPERATIONS = 732
MAX_INPUT_BYTES = 16 * 1024**2


@dataclass(frozen=True)
class RetirementOperation:
    _encoded: bytes
    state: str

    @property
    def inputs(self):
        return json.loads(self._encoded)


def _same_context(data, anchor):
    inputs = data.get("inputs", {})
    expected = anchor.inputs
    return (
        data.get("kind") == FEATURE_KIND
        and type(inputs) is dict
        and type(inputs.get("source")) is dict
        and inputs.get("source", {}).get("report_sha256") == expected["report_sha256"]
        and inputs.get("origin") == expected["origin"]
        and inputs.get("coins") == expected["coins"]
        and inputs.get("semantics") == expected["semantics"]
        and inputs.get("execution_policy") == expected.get("execution_policy")
    )


def owner_context(resources, db, owner):
    """Return authenticated anchor metadata without opening its payload."""
    if not _digest(owner):
        raise ValueError("Invalid rolling retirement owner metadata")
    row = db.execute(
        "SELECT key, CASE WHEN length(CAST(descriptor AS BLOB)) BETWEEN 1 AND ? "
        "THEN descriptor END, sha256 FROM publications WHERE key=?",
        (MAX_DESCRIPTOR_BYTES, owner),
    ).fetchone()
    if row is None:
        raise ValueError("Rolling retirement owner metadata is missing")
    _descriptor(resources, db, *row)
    data = json.loads(row[1])
    if data["kind"] != ANCHOR_KIND:
        raise ValueError("Rolling retirement owner is not an anchor")
    return data["inputs"]


def _relevant(resources, db, receipt, anchor):
    """Use authenticated owner metadata; never open unrelated journal payloads."""
    owner = receipt["owner"]
    if owner == anchor.publication.key:
        return True
    data = owner_context(resources, db, owner)
    expected = anchor.inputs
    return all(
        data.get(key) == expected.get(key)
        for key in (
            "report_sha256",
            "origin",
            "coins",
            "semantics",
            "execution_policy",
        )
    )


def _target_binding(data, anchor, report_pin):
    inputs, day, cutoff = feature_context(data["inputs"])
    if cutoff > anchor.first:
        raise ValueError("Rolling retirement crosses retained anchor boundary")
    source = QualifiedWindow(report_pin, day, cutoff)
    if inputs["source"] != source.inputs():
        raise ValueError("Rolling retirement source membership mismatch")
    source.verify()


def _state(resources, db, inputs, body, publication):
    committed = PublishedArtifacts(resources)._read(
        db, publication_key(COMMITTED, inputs), COMMITTED, inputs
    )
    if committed is not None:
        targets = _detached(resources, db, body, inputs, publication)
        return "detached" if any(pending for _, _, pending in targets) else "complete"
    row = db.execute(
        "SELECT descriptor FROM publications WHERE key=?", (inputs["target"],)
    ).fetchone()
    if row != (body["inventory"]["descriptor"],):
        raise ValueError("Uncommitted retirement target changed or disappeared")
    for allocation in body["allocations"]:
        if db.execute(
            "SELECT * FROM allocations WHERE token=?", (allocation[0],)
        ).fetchone() != tuple(allocation):
            raise ValueError("Prepared retirement allocation changed")
    _files(resources, body)
    return "prepared"


def inspect_owned(resources, report_pin, anchor, check):
    """Authenticate all records before allowing any selected-owner mutation."""
    namespace = _namespace(resources)
    result, unresolved, encoded_bytes = [], set(), 0
    owner = anchor.publication.key
    with (
        pin_directory(resources.root, namespace[0]),
        pin_directory(resources.root / "artifacts", namespace[1]),
        _PinnedFile(resources.path, METADATA_BYTES // 2) as database,
        _PinnedFile(resources.marker, 4096) as marker,
    ):
        with resources._connect() as db:
            version = db.execute("PRAGMA data_version").fetchone()
            for row in _rows(db):
                _descriptor(resources, db, *row)
                data = json.loads(row[1])
                if (
                    data["kind"] not in (INTENT, COMMITTED)
                    or "owner" not in data["inputs"]
                ):
                    continue
                receipt = data["inputs"]
                if (
                    data["kind"] == COMMITTED
                    and db.execute(
                        "SELECT 1 FROM publications WHERE key=?",
                        (publication_key(INTENT, receipt),),
                    ).fetchone()
                ):
                    continue
                if not _relevant(resources, db, receipt, anchor):
                    continue
                body, publication = load_journal(resources, receipt)
                target = json.loads(body["inventory"]["descriptor"])
                if not _same_context(target, anchor):
                    if receipt["owner"] == owner:
                        raise ValueError(
                            "Rolling retirement owner target context mismatch"
                        )
                    continue
                state = _state(resources, db, receipt, body, publication)
                if receipt["owner"] != owner:
                    if state != "complete":
                        raise ValueError(
                            "Unresolved rolling retirement has another owner"
                        )
                    continue
                _target_binding(target, anchor, report_pin)
                if state != "complete":
                    if receipt["target"] in unresolved:
                        raise ValueError(
                            "Multiple ambiguous unresolved retirement intents"
                        )
                    unresolved.add(receipt["target"])
                encoded = _encode(receipt)
                encoded_bytes += len(encoded)
                if len(result) >= MAX_OPERATIONS or encoded_bytes > MAX_INPUT_BYTES:
                    raise ValueError(
                        "Rolling retirement inventory resource limit exceeded"
                    )
                result.append(RetirementOperation(encoded, state))
                check()
            anchor.verify()
            check()
            if db.execute("PRAGMA data_version").fetchone() != version:
                raise ValueError(
                    "Rolling retirement catalogue changed during inventory"
                )
            database.check()
            marker.check()
    return tuple(result)
