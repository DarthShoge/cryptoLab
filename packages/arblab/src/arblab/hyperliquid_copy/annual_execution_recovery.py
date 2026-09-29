"""Explicit recovery for a previously journaled annual ranking retirement."""

from .derived_publication import ArtifactPin, Publication, PublishedArtifacts
from .disk_score_result import read_cohort
from .annual_ranking_retirement import _load_operation, recover_operation


def _persisted_saved_reload(resources, body):
    """Reload the exact saved ranking using only immutable journal authority."""
    saved = body["targets"][1]
    artifacts = tuple(
        ArtifactPin(row[0], row[1], row[5], row[6]) for row in saved["allocations"]
    )
    publication = Publication(saved["key"], artifacts)
    live = PublishedArtifacts(resources).lookup(saved["kind"], saved["inputs"])
    if live != publication:
        raise ValueError("Journaled saved ranking publication changed")
    cohort = read_cohort(
        resources.root, publication, saved["inputs"]["query"]["selection"]
    )
    consumption = body["consumption"]
    if (
        cohort.candidate_count != consumption["candidate_count"]
        or cohort.eligible_count != consumption["eligible_count"]
        or len(cohort.selected) != consumption["selected_count"]
        or cohort.artifact.sha256 != consumption["artifact_sha256"]
        or tuple(pin.token for pin in artifacts)
        != tuple(consumption["artifact_tokens"])
    ):
        raise ValueError("Journaled saved ranking reload changed")


def operation_status(resources, operation_inputs):
    frozen, _, body = _load_operation(resources, operation_inputs)
    keys = [row["key"] for row in body["targets"]]
    with resources._connect() as db:
        present = {
            key: db.execute("SELECT 1 FROM publications WHERE key=?", (key,)).fetchone()
            is not None
            for key in keys
        }
    values = list(present.values())
    if values == [True, True, True]:
        state = "prepared"
    elif values == [False, True, True]:
        state = "candidate_detached"
    elif values == [False, False, True]:
        state = "saved_detached"
    elif values == [False, False, False]:
        state = "targets_detached"
    else:
        raise ValueError("Unrecognized annual retirement recovery state")
    return dict(operation=frozen, state=state, targets=present)


def recover(resources, operation_inputs, *, verify_saved=None):
    status = operation_status(resources, operation_inputs)
    _, _, body = _load_operation(resources, operation_inputs)
    if status["state"] in ("prepared", "candidate_detached"):
        # Reject changed saved evidence before the first new mutation, then
        # repeat this check after candidate disposal inside execute_operation.
        _persisted_saved_reload(resources, body)

        def persisted():
            _persisted_saved_reload(resources, body)
            if verify_saved is not None:
                verify_saved()

        verifier = persisted
    else:
        verifier = verify_saved
    result = recover_operation(resources, operation_inputs, verify_saved=verifier)
    return dict(
        before=status,
        result=result,
        after=operation_status(resources, operation_inputs),
    )
