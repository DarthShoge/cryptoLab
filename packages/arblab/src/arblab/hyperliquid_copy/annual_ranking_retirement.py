"""Read-only discovery of the exact annual ranking publications to retire."""

from dataclasses import dataclass
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

from .annual_execution_policy import POLICY
from .annual_ranking_consumption import RankingConsumption
from .bound_scoring_context import _read_publication
from .cache_retirement_inventory import _descriptor, _rows
from .cache_retirement import _unlink_owned
from .cache_retirement_inventory import _identity
from .derived_publication import PublishedArtifacts, _encode, _inputs, publication_key
from .annual_execution_policy import ANNUAL_BOUNDED_ROLLING
from .download import file_hash
from .ranking_staging_binding import KIND as STAGED_KIND
from .ranking_staging_policy import POLICY as STAGING_POLICY
from .saved_ranking_lookup import KIND as SAVED_KIND

INTENT = "annual_ranking_retirement_intent"
COMMITTED = "annual_ranking_retirement_committed"
MAX_OPERATION_BYTES = 64 * 1024


@dataclass(frozen=True)
class RetirementTarget:
    kind: str
    inputs: dict
    key: str
    tokens: tuple[str, ...]
    artifact_sha256: tuple[str, ...]


@dataclass(frozen=True)
class AnnualRankingTargets:
    candidate: RetirementTarget
    saved: RetirementTarget
    staged: RetirementTarget

    @property
    def ordered(self):
        return self.candidate, self.saved, self.staged


def _target(publication, kind, inputs):
    if publication.key != publication_key(kind, inputs):
        raise ValueError("Annual retirement publication identity mismatch")
    return RetirementTarget(
        kind=kind,
        inputs=inputs,
        key=publication.key,
        tokens=tuple(pin.token for pin in publication.artifacts),
        artifact_sha256=tuple(pin.sha256 for pin in publication.artifacts),
    )


def discover_targets(resources, saved_inputs, consumption):
    """Authenticate three invocation-owned targets without changing the cache."""
    resources.lease.check()
    if (
        not isinstance(consumption, RankingConsumption)
        or consumption.execution_policy != POLICY
        or type(saved_inputs) is not dict
        or not re.fullmatch("[a-f0-9]{64}", consumption.config_sha256)
    ):
        raise ValueError("Invalid annual ranking retirement authorization")
    saved_publication, saved = _read_publication(
        resources, consumption.ranking_key, SAVED_KIND
    )
    if _encode(saved) != _encode(saved_inputs):
        raise ValueError("Consumed saved ranking inputs changed")
    staged_publication, staged = _read_publication(
        resources, saved["ranking"], STAGED_KIND
    )
    try:
        candidate_inputs = staged["candidate"]
        candidate_key = publication_key("candidate_history", candidate_inputs)
        candidate_publication, candidate = _read_publication(
            resources, candidate_key, "candidate_history"
        )
        query = staged["query"]
        source = query["source"]
        if (
            staged["policy"] != STAGING_POLICY
            or query["execution_policy"] != POLICY
            or candidate["execution_policy"] != POLICY
            or candidate["max_bytes"] != 64 * 1024**2
            or candidate["candidate_day_bytes"] != 8 * 1024**2
            or source["pin"]["path"] != consumption.source_path
            or source["pin"]["sha256"] != consumption.source_sha256
            or staged["summary"]["candidate_count"] != consumption.candidate_count
            or staged["summary"]["eligible_count"] != consumption.eligible_count
            or staged["summary"]["selected_count"] != consumption.selected_count
            or query["metrics"]["decision_time"] != consumption.decision.isoformat()
            or query["metrics"]["scope"] != consumption.scope
        ):
            raise ValueError("Annual ranking retirement provenance mismatch")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed annual ranking retirement provenance") from exc
    if _encode(candidate) != _encode(candidate_inputs):
        raise ValueError("Candidate history binding changed")

    targets = AnnualRankingTargets(
        candidate=_target(candidate_publication, "candidate_history", candidate),
        saved=_target(saved_publication, SAVED_KIND, saved),
        staged=_target(staged_publication, STAGED_KIND, staged),
    )
    if (
        targets.saved.tokens != consumption.artifact_tokens
        or targets.saved.artifact_sha256 != (consumption.artifact_sha256,)
        or targets.staged.tokens != targets.saved.tokens
        or targets.staged.artifact_sha256 != targets.saved.artifact_sha256
        or set(targets.candidate.tokens) & set(targets.saved.tokens)
    ):
        raise ValueError("Annual ranking retirement artifact ownership mismatch")

    allowed = {
        token: {target.key for target in targets.ordered if token in target.tokens}
        for target in targets.ordered
        for token in target.tokens
    }
    with resources._connect() as db:
        actual = {token: set() for token in allowed}
        for row in _rows(db):
            publication = _descriptor(resources, db, *row)
            for pin in publication.artifacts:
                if pin.token in actual:
                    actual[pin.token].add(publication.key)
    if actual != allowed:
        raise ValueError("Unexpected reference to annual retirement artifact")
    resources.lease.check()
    return targets


def _engine():
    return file_hash(Path(__file__))


def _consumption(value):
    result = asdict(value)
    result["decision"] = value.decision.isoformat()
    result["artifact_tokens"] = list(value.artifact_tokens)
    return result


def _operation_body(resources, targets, consumption):
    rows = []
    with resources._connect() as db:
        for target in targets.ordered:
            allocation_rows, identities = [], []
            for token in target.tokens:
                row = db.execute(
                    "SELECT * FROM allocations WHERE token=?", (token,)
                ).fetchone()
                if row is None:
                    raise ValueError("Annual retirement allocation is missing")
                allocation_rows.append(list(row))
                identities.append([token, list(_identity(resources.root / row[1]))])
            rows.append(
                dict(
                    kind=target.kind,
                    inputs=target.inputs,
                    key=target.key,
                    tokens=list(target.tokens),
                    artifact_sha256=list(target.artifact_sha256),
                    allocations=allocation_rows,
                    identities=identities,
                )
            )
    return dict(
        schema=1,
        policy=POLICY,
        consumption=_consumption(consumption),
        targets=rows,
        engine=_engine(),
    )


def prepare_operation(resources, saved_inputs, consumption):
    """Publish one bounded operation journal before the first target detaches."""
    targets = discover_targets(resources, saved_inputs, consumption)
    body = _operation_body(resources, targets, consumption)
    raw = _encode(body)
    if len(raw) > MAX_OPERATION_BYTES:
        raise ValueError("Annual retirement operation byte limit exceeded")
    inputs = _inputs(
        INTENT,
        dict(
            schema=1,
            policy=POLICY,
            operation_sha256=hashlib.sha256(raw).hexdigest(),
            targets=[target.key for target in targets.ordered],
            engine=_engine(),
        ),
    )
    publications = PublishedArtifacts(resources)
    existing = publications.lookup(INTENT, inputs)
    if existing is not None:
        return inputs
    relative = "artifacts/" + uuid.uuid4().hex
    token = resources.reserve(relative, MAX_OPERATION_BYTES, "payload")
    path = resources.root / relative
    with path.open("xb") as output:
        output.write(raw)
        output.flush()
        os.fsync(output.fileno())
    resources.settle(token)
    with resources._connect() as db:
        pins = PublishedArtifacts(resources)._records(db, [token])
    for kind in (INTENT, COMMITTED):
        descriptor = _encode(
            dict(
                schema=1,
                kind=kind,
                inputs=inputs,
                artifacts=[asdict(pin) for pin in pins],
            )
        )
        if len(descriptor) > ANNUAL_BOUNDED_ROLLING.operation_descriptor_bytes:
            raise ValueError("Annual retirement publication descriptor limit exceeded")
    publications.publish(INTENT, inputs, [token])
    return inputs


def _load_operation(resources, inputs):
    frozen = _inputs(INTENT, inputs)
    if frozen.get("engine") != _engine() or frozen.get("policy") != POLICY:
        raise ValueError("Annual retirement operation engine/policy changed")
    publications = PublishedArtifacts(resources)
    publication = publications.lookup(INTENT, frozen)
    if publication is None:
        publication = publications.lookup(COMMITTED, frozen)
    if publication is None or len(publication.artifacts) != 1:
        raise ValueError("Annual retirement operation is missing")
    pin = publication.artifacts[0]
    if not 0 < pin.bytes <= MAX_OPERATION_BYTES:
        raise ValueError("Annual retirement operation size changed")
    raw = (resources.root / pin.path).read_bytes()
    try:
        body = json.loads(raw)
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("Malformed annual retirement operation") from exc
    if (
        _encode(body) != raw
        or hashlib.sha256(raw).hexdigest() != frozen.get("operation_sha256")
        or body.get("schema") != 1
        or body.get("policy") != POLICY
        or body.get("engine") != _engine()
        or [row.get("key") for row in body.get("targets", [])] != frozen.get("targets")
    ):
        raise ValueError("Annual retirement operation binding changed")
    return frozen, publication, body


def _references(resources, db, tokens):
    result = {token: set() for token in tokens}
    for row in _rows(db):
        publication = _descriptor(resources, db, *row)
        for pin in publication.artifacts:
            if pin.token in result:
                result[pin.token].add(publication.key)
    return result


def _detach(resources, operation_inputs, operation_key, row, remaining_keys):
    tokens = tuple(row["tokens"])
    with resources._connect() as db, db:
        db.execute("BEGIN IMMEDIATE")
        resources._audit(db)
        current = db.execute(
            "SELECT descriptor FROM publications WHERE key=?", (row["key"],)
        ).fetchone()
        if current is None:
            # The saved receipt deliberately shares its artifacts with the staged
            # publication; only the final staged detach owns their disposal.
            if row["kind"] == SAVED_KIND:
                return ()
            pending = []
            for allocation, identity in zip(
                row["allocations"], row["identities"], strict=True
            ):
                current_allocation = db.execute(
                    "SELECT * FROM allocations WHERE token=?", (allocation[0],)
                ).fetchone()
                if current_allocation is None:
                    continue
                if current_allocation[1:4] != tuple(
                    allocation[1:4]
                ) or current_allocation[4] not in ("pending", "retained"):
                    raise ValueError("Annual retirement allocation changed")
                if current_allocation[4] == "pending":
                    if current_allocation[5:] != (None, None):
                        raise ValueError("Invalid pending annual retirement allocation")
                    pending.append((allocation, tuple(identity[1])))
                elif row["kind"] != SAVED_KIND:
                    raise ValueError(
                        "Detached annual retirement allocation remained retained"
                    )
            resources._audit(db)
            return tuple(pending)
        descriptor = json.loads(current[0])
        expected_artifacts = [
            dict(token=value[0], path=value[1], bytes=value[5], sha256=value[6])
            for value in row["allocations"]
        ]
        if _encode(descriptor) != _encode(
            dict(
                schema=1,
                kind=row["kind"],
                inputs=row["inputs"],
                artifacts=expected_artifacts,
            )
        ) or tuple(item["sha256"] for item in expected_artifacts) != tuple(
            row["artifact_sha256"]
        ):
            raise ValueError("Annual retirement target changed")
        allowed = set(remaining_keys) | {row["key"], operation_key}
        references = _references(resources, db, tokens)
        if any(keys - allowed for keys in references.values()):
            raise ValueError("Unexpected reference to annual retirement artifact")
        db.execute("DELETE FROM publications WHERE key=?", (row["key"],))
        references = _references(resources, db, tokens)
        pending = []
        for allocation, identity in zip(
            row["allocations"], row["identities"], strict=True
        ):
            token = allocation[0]
            if not references[token]:
                db.execute(
                    "UPDATE allocations SET maximum=?,state='pending',bytes=NULL,sha256=NULL WHERE token=?",
                    (allocation[2], token),
                )
                pending.append((allocation, tuple(identity[1])))
        resources._audit(db)
    return tuple(pending)


def _dispose(resources, operation_inputs, pending):
    if not pending:
        return
    directory_fd = os.open(
        resources.root / "artifacts", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    try:
        for row, identity in pending:

            def guard():
                _load_operation(resources, operation_inputs)

            _unlink_owned(resources, directory_fd, row, identity, guard)
            resources.release_missing(row[0])
    finally:
        os.close(directory_fd)


def execute_operation(resources, operation_inputs, *, verify_saved):
    """Detach candidate, verify saved reload, then detach saved and staged."""
    if verify_saved is not None and not callable(verify_saved):
        raise ValueError("Saved ranking reload callback required")
    frozen, publication, body = _load_operation(resources, operation_inputs)
    targets = body["targets"]
    if [row.get("kind") for row in targets] != [
        "candidate_history",
        SAVED_KIND,
        STAGED_KIND,
    ]:
        raise ValueError("Invalid annual retirement target order")
    operation_key = publication.key
    pending = _detach(
        resources,
        frozen,
        operation_key,
        targets[0],
        [row["key"] for row in targets[1:]],
    )
    _dispose(resources, frozen, pending)
    with resources._connect() as db:
        saved_exists = (
            db.execute(
                "SELECT 1 FROM publications WHERE key=?", (targets[1]["key"],)
            ).fetchone()
            is not None
        )
    if saved_exists:
        if verify_saved is None:
            raise ValueError("Saved ranking reload callback required")
        verify_saved()
    pending = _detach(resources, frozen, operation_key, targets[1], [targets[2]["key"]])
    if pending:
        raise ValueError("Saved ranking detached the shared artifact too early")
    pending = _detach(resources, frozen, operation_key, targets[2], [])
    _dispose(resources, frozen, pending)
    committed = PublishedArtifacts(resources).publish(
        COMMITTED, frozen, [publication.artifacts[0].token]
    )
    with resources._connect() as db, db:
        db.execute("BEGIN IMMEDIATE")
        if (
            PublishedArtifacts(resources)._read(db, committed.key, COMMITTED, frozen)
            != committed
        ):
            raise ValueError("Annual retirement commit changed")
        db.execute(
            "DELETE FROM publications WHERE key=?",
            (publication_key(INTENT, frozen),),
        )
        resources._audit(db)
    return dict(
        operation=publication_key(COMMITTED, frozen),
        retired_targets=[row["key"] for row in targets],
    )


def recover_operation(resources, operation_inputs, *, verify_saved=None):
    """Resume only the exact journaled operation; ordinary cache open does nothing."""
    return execute_operation(resources, operation_inputs, verify_saved=verify_saved)
