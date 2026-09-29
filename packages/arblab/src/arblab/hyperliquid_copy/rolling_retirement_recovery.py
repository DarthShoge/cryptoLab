"""Explicit recovery only; never chooses new targets or creates retirement intents."""

from pathlib import Path

from .cache_retirement import begin_retirement, finish_retirement
from .cache_retirement_journal import engine as retirement_engine
from .derived_publication import _encode, _inputs
from .download import file_hash
from .feature_resume_anchor import FeatureAnchor, KIND, _engine as anchor_engine
from .rolling_retirement_inventory import inspect_owned
from .rolling_anchor_guard import AnchorGuard


def _engine():
    root = Path(__file__).parent
    return dict(
        code={
            name: file_hash(root / name)
            for name in (
                "rolling_retirement_inventory.py",
                "rolling_retirement_recovery.py",
                "rolling_anchor_guard.py",
                "qualified_source_session.py",
            )
        },
        retirement=retirement_engine(),
        anchor=anchor_engine(),
    )


def _binding(resources, report_pin, anchor_inputs, validation=None):
    resources.lease.check()
    pin_encoded = _encode(report_pin)
    encoded = _encode(_inputs(KIND, anchor_inputs))
    engine = _encode(_engine())
    anchor = FeatureAnchor(resources, report_pin, anchor_inputs)

    def check():
        if validation is not None:
            validation()
        if (
            _encode(_engine()) != engine
            or _encode(report_pin) != pin_encoded
            or _encode(anchor_inputs) != encoded
        ):
            raise ValueError("Rolling retirement context changed")
        if validation is not None:
            validation()
        resources.lease.check()

    check()
    return anchor, check


def inspect_feature_retirements(resources, report_pin, anchor_inputs):
    """Read-only inventory; foreign unresolved owners fail before any mutation."""
    anchor, check = _binding(resources, report_pin, anchor_inputs)
    protection = AnchorGuard(resources, report_pin, anchor, check)
    result = inspect_owned(resources, report_pin, anchor, protection.check)
    protection.check()
    return result


def recover_feature_retirements(
    resources, report_pin, anchor_inputs, *, validation=None
):
    """Resume only this anchor's authenticated operations; do not retry failures."""
    anchor, check = _binding(resources, report_pin, anchor_inputs, validation)
    protection = AnchorGuard(resources, report_pin, anchor, check)
    operations = inspect_owned(resources, report_pin, anchor, protection.check)
    for operation in operations:
        anchor.verify()
        protection.check()
        if operation.state == "prepared":
            begin_retirement(resources, operation.inputs, validation=protection.check)
        if operation.state != "complete":
            finish_retirement(resources, operation.inputs, validation=protection.check)
        anchor.verify()
        protection.check()
    return tuple(operation.inputs["target"] for operation in operations)
