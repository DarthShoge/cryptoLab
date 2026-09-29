"""Explicit rolling feature policy; existing non-retiring readers remain unchanged."""

from datetime import timedelta
import json
from pathlib import Path

from .cache_retirement import prepare_retirement, begin_retirement, finish_retirement
from .annual_execution_policy import execution_policy
from .contracts import utc
from .derived_publication import _encode, _inputs
from .download import file_hash
from .feature_history import FeatureHistory, _engine as history_engine
from .feature_resume_anchor import publish_feature_anchor
from .rolling_anchor_guard import AnchorGuard
from .rolling_feature_catalogue import (
    select_anchor,
    require_no_pending,
    expired_feature_days,
)
from .rolling_retirement_recovery import (
    recover_feature_retirements,
    _engine as recovery_engine,
)
from .qualified_source_session import QualifiedSourceSession


def _engine():
    root = Path(__file__).parent
    return dict(
        code={
            name: file_hash(root / name)
            for name in ("rolling_feature_history.py", "rolling_feature_catalogue.py")
        },
        history=history_engine(),
        recovery=recovery_engine(),
    )


class RollingFeatureHistory:
    def __init__(
        self,
        resources,
        report_pin,
        coins,
        semantics,
        *,
        execution_policy_name=None,
    ):
        self.resources = self._resources = resources
        self._pin, self._coins, self._semantics = report_pin, coins, semantics
        self._execution_policy = execution_policy_name
        self._annual_policy = execution_policy(execution_policy_name)
        self._frozen = self._context()
        self._engine = _encode(_engine())
        if self._context() != self._frozen:
            raise ValueError("Rolling history caller context changed")
        self._source = FeatureHistory(
            resources,
            report_pin,
            coins,
            semantics,
            execution_policy_name=execution_policy_name,
        )
        self._session = QualifiedSourceSession(report_pin)
        self._last_start = self._last_end = None
        self._anchor = None
        self._busy = False
        self._validation = None
        self._check()

    @property
    def anchor_inputs(self):
        return None if self._anchor is None else json.loads(self._anchor)

    def _context(self):
        return _encode(
            _inputs(
                "rolling_feature_history",
                dict(
                    pin=self._pin,
                    coins=self._coins,
                    semantics=self._semantics,
                    **(
                        {"execution_policy": self._execution_policy}
                        if self._execution_policy is not None
                        else {}
                    ),
                ),
            )
        )

    def _check(self):
        if self.resources is not self._resources or _encode(_engine()) != self._engine:
            raise ValueError("Rolling history context changed")
        self._source._verify()
        self._session.verify()
        if self._validation is not None:
            self._validation()
        if self._context() != self._frozen:
            raise ValueError("Rolling history caller context changed")
        self._session.verify_identity()
        self._resources.lease.check()

    def window(self, start, end, *, validation=None):
        if self._busy:
            raise ValueError("Rolling history is busy")
        if validation is not None and not callable(validation):
            raise ValueError("Invalid rolling history validation")
        self._busy = True
        self._validation = validation
        try:
            self._check()
            start, end = utc(start), utc(end)
            if (
                not self._source.origin <= start < end <= self._source.finish
                or end - start > timedelta(days=732)
            ):
                raise ValueError("Invalid bounded rolling history interval")
            if (
                self._last_start is not None
                and start < self._last_start
                or self._last_end is not None
                and end < self._last_end
            ):
                raise ValueError("Rolling history decision/lookback reversed")
            window, anchor, protection = self._advance(start, end)
            window.verify()
            self._check()
            protection.check()
            self._anchor = _encode(anchor.inputs)
            self._last_start, self._last_end = start, end
            return window
        finally:
            self._validation = None
            self._busy = False

    def _advance(self, start, end):
        resources, pin = self._resources, self._pin
        anchor = select_anchor(
            resources,
            pin,
            self._coins,
            self._semantics,
            start,
            end,
            validation=self._check,
            execution_policy_name=self._execution_policy,
        )
        if anchor is None:
            require_no_pending(
                resources,
                pin,
                self._coins,
                self._semantics,
                validation=self._check,
                execution_policy_name=self._execution_policy,
            )
        else:
            recover_feature_retirements(
                resources, pin, anchor.inputs, validation=self._check
            )
        self._check()
        history = FeatureHistory(
            resources,
            pin,
            self._coins,
            self._semantics,
            anchor_inputs=None if anchor is None else anchor.inputs,
            execution_policy_name=self._execution_policy,
        )
        window = history.window(start, end)
        self._check()
        anchor = publish_feature_anchor(
            resources, pin, window.days, self._coins, self._semantics
        )
        self._check()
        resumed = FeatureHistory(
            resources,
            pin,
            self._coins,
            self._semantics,
            anchor_inputs=anchor.inputs,
            execution_policy_name=self._execution_policy,
        ).window(start, end)
        resumed.verify()
        if resumed.inputs() != window.inputs():
            raise ValueError("Replacement anchor changed the active feature window")
        window = resumed
        self._check()
        recover_feature_retirements(
            resources, pin, anchor.inputs, validation=self._check
        )
        protection = AnchorGuard(resources, pin, anchor, self._check)
        targets = expired_feature_days(
            resources,
            pin,
            anchor,
            validation=protection.check,
            execution_policy_name=self._execution_policy,
        )
        protected = [
            anchor.publication.key,
            *(day.publication.key for day in anchor.days),
        ]
        for day in targets:
            protection.check()
            receipt = prepare_retirement(
                resources,
                "qualified_features_day",
                day.inputs,
                protected_keys=protected,
                owner=anchor.publication.key,
                reason="Obsolete complete feature day before verified rolling anchor",
                max_bytes=(
                    self._annual_policy.retirement_journal_bytes
                    if self._annual_policy is not None
                    else 4 * 1024**2
                ),
                max_descriptor_bytes=(
                    self._annual_policy.retirement_descriptor_bytes
                    if self._annual_policy is not None
                    else 1024**2
                ),
            )
            protection.check()
            begin_retirement(resources, receipt, validation=protection.check)
            finish_retirement(resources, receipt, validation=protection.check)
            anchor.verify()
            protection.check()
        return window, anchor, protection
