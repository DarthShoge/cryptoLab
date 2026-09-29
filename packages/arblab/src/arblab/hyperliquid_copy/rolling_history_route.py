"""Resolve pending retirement before choosing a complete ranking producer."""

from datetime import timedelta
from pathlib import Path

from .contracts import utc
from .derived_publication import _encode, _inputs
from .download import file_hash
from .qualified_source_session import QualifiedSourceSession
from .rolling_anchor_guard import AnchorGuard
from .rolling_feature_catalogue import select_anchor, require_no_pending
from .rolling_retirement_recovery import (
    recover_feature_retirements,
    _engine as recovery_engine,
)


def _engine():
    root = Path(__file__).parent
    return dict(
        code={
            name: file_hash(root / name)
            for name in (
                "rolling_history_route.py",
                "rolling_feature_catalogue.py",
            )
        },
        recovery=recovery_engine(),
    )


def prepare_history_route(
    resources,
    source_session,
    coins,
    semantics,
    start,
    end,
    *,
    validation=None,
    execution_policy_name=None,
):
    if type(source_session) is not QualifiedSourceSession:
        raise ValueError("Verified source session required for history routing")
    if validation is not None and not callable(validation):
        raise ValueError("Invalid history route validation")
    start, end = utc(start), utc(end)

    def context():
        return _encode(
            _inputs(
                "rolling_history_route",
                dict(
                    source=source_session.inputs(),
                    coins=coins,
                    semantics=semantics,
                    start=start.isoformat(),
                    end=end.isoformat(),
                    execution_policy=execution_policy_name,
                ),
            )
        )

    frozen = context()
    engine = _encode(_engine())

    def check():
        resources.lease.check()
        source_session.verify()
        if _encode(_engine()) != engine:
            raise ValueError("History route engine changed")
        if validation is not None:
            validation()
        if context() != frozen:
            raise ValueError("History route context changed")
        source_session.verify_identity()
        resources.lease.check()

    check()
    if (
        type(coins) not in (list, tuple)
        or list(coins) != sorted(source_session.coins)
        or semantics not in ("gross_excludes_fee", "net_includes_fee")
        or not source_session.origin <= start < end <= source_session.finish
        or end - start > timedelta(days=732)
    ):
        raise ValueError("Invalid history route source scope or interval")
    pin = source_session.inputs()["pin"]
    anchor = select_anchor(
        resources,
        pin,
        coins,
        semantics,
        source_session.finish - timedelta(microseconds=1),
        source_session.finish,
        validation=check,
        execution_policy_name=execution_policy_name,
    )
    if anchor is None:
        require_no_pending(
            resources,
            pin,
            coins,
            semantics,
            validation=check,
            execution_policy_name=execution_policy_name,
        )
        return "features"
    protection = AnchorGuard(resources, pin, anchor, check)
    recover_feature_retirements(
        resources, pin, anchor.inputs, validation=protection.check
    )
    first = start.replace(hour=0, minute=0, second=0, microsecond=0)
    last = end.replace(hour=0, minute=0, second=0, microsecond=0)
    if last < end:
        last += timedelta(days=1)
    result = "raw" if first < anchor.first or last < anchor.cutoff else "features"
    anchor.verify()
    protection.check()
    return result
