"""Chronological feature-day orchestration; no eviction or implicit recovery."""

from datetime import timedelta
from pathlib import Path

from .archive_cache import _safe
from .annual_execution_policy import execution_policy
from .contracts import symbol, utc
from .derived_publication import _encode, _inputs
from .download import file_hash
from .feature_day_builder import build_feature_day
from .feature_publication import feature_engine
from .feature_resume_anchor import (
    FeatureAnchor,
    KIND as ANCHOR_KIND,
    _engine as anchor_engine,
)
from .feature_window import FeatureWindow
from .prefix_qualification import _previous, _engine as qualification_engine
from .qualified_day import _day

MAX_DAYS = 733


def _engine():
    root = Path(__file__).parent
    return dict(
        code={
            name: file_hash(root / name)
            for name in (
                "feature_history.py",
                "feature_window.py",
            )
        },
        features=feature_engine(),
        anchor=anchor_engine(),
        qualification=qualification_engine(),
    )


class FeatureHistory:
    def __init__(
        self,
        resources,
        report_pin,
        coins,
        semantics,
        *,
        anchor_inputs=None,
        execution_policy_name=None,
    ):
        resources.lease.check()
        frozen_anchor = _anchor_context(anchor_inputs)
        initial_engine = _engine()
        if type(report_pin) is not dict or set(report_pin) != {"path", "sha256"}:
            raise ValueError("Pinned feature history report required")
        report = _previous(report_pin, qualification_engine())
        if (
            type(coins) not in (list, tuple)
            or not 1 <= len(coins) <= 50
            or list(coins) != sorted({symbol(c) for c in coins})
            or not set(coins) <= set(report["coins"])
            or semantics not in ("gross_excludes_fee", "net_includes_fee")
        ):
            raise ValueError("Invalid feature history scope/semantics")
        self.resources = resources
        self._resources = resources
        self._caller_pin, self._caller_coins = report_pin, coins
        self.pin = dict(report_pin)
        self.coins = tuple(coins)
        self.semantics = semantics
        policy = execution_policy(execution_policy_name)
        self.execution_policy = None if policy is None else policy.name
        self.origin = _day(report["source_start"])
        self.finish = _day(report["source_end"])
        self._days = ()
        self._base = self.origin
        self._caller_anchor = anchor_inputs
        self._anchor_inputs = frozen_anchor
        if self._anchor_inputs is not None:
            anchor = FeatureAnchor(resources, report_pin, self._anchor_inputs)
            if (
                anchor.inputs["coins"] != list(self.coins)
                or anchor.inputs["semantics"] != semantics
                or anchor.inputs["origin"] != self.origin.date().isoformat()
                or len(anchor.days) > MAX_DAYS
            ):
                raise ValueError("Feature history anchor context/limit mismatch")
            self._base, self._days = anchor.first, anchor.days
        self._last_end = None
        self._busy = False
        self._frozen = self._context()
        if _engine() != initial_engine:
            raise ValueError("Feature history engine changed during construction")
        self._verify()

    def _context(self):
        return _encode(
            dict(
                pin=self.pin,
                coins=self.coins,
                semantics=self.semantics,
                origin=self.origin.isoformat(),
                finish=self.finish.isoformat(),
                base=self._base.isoformat(),
                anchor=self._anchor_inputs,
                engine=_engine(),
                **(
                    {"execution_policy": self.execution_policy}
                    if self.execution_policy is not None
                    else {}
                ),
            )
        )

    def _check_context(self):
        encoded = self._context()
        self._resources.lease.check()
        if (
            self.resources is not self._resources
            or self._caller_pin != self.pin
            or tuple(self._caller_coins) != self.coins
            or _encode(_anchor_context(self._caller_anchor))
            != _encode(self._anchor_inputs)
            or encoded != self._frozen
        ):
            raise ValueError("Feature history context changed")

    def _verify(self):
        self._check_context()
        path = Path(self.pin["path"])
        _safe(path)
        if file_hash(path) != self.pin["sha256"]:
            raise ValueError("Feature history source/context changed")
        self._check_context()

    def window(self, start, end):
        if self._busy:
            raise ValueError("Feature history is busy")
        self._verify()
        start, end = utc(start), utc(end)
        if self._last_end is not None and end < self._last_end:
            raise ValueError("Feature history decision reversed")
        if not self.origin <= start < end <= self.finish or end - start > timedelta(
            days=732
        ):
            raise ValueError("Invalid bounded feature history interval")
        first = start.replace(hour=0, minute=0, second=0, microsecond=0)
        last = end.replace(hour=0, minute=0, second=0, microsecond=0)
        if last != end:
            last += timedelta(days=1)
        if first < self._base:
            raise ValueError("Requested lookback precedes retained feature history")
        if (last - self.origin).days > MAX_DAYS:
            raise ValueError("Feature history descriptor limit exceeded")
        if max(len(self._days), (last - self._base).days) > MAX_DAYS:
            raise ValueError("Retained feature history descriptor limit exceeded")
        self._busy = True
        try:
            result, days = self._advance(first, last, start, end)
            self._verify()
            self._days, self._last_end = tuple(days), end
            return result
        finally:
            self._busy = False

    def _advance(self, first, last, start, end):
        days = list(self._days)
        at = days[-1].cutoff if days else self.origin
        while at < last:
            self._verify()
            previous = days[-1] if days else None
            result = build_feature_day(
                self.resources,
                self.pin,
                at,
                self.coins,
                self.semantics,
                previous,
                execution_policy_name=self.execution_policy,
            )
            self._verify()
            days.append(result)
            at = result.cutoff
        result = FeatureWindow(
            self.resources,
            self.pin,
            days[(first - self._base).days : (last - self._base).days],
            start,
            end,
            self.coins,
            self.semantics,
        )
        return result, days


def _anchor_context(value):
    return None if value is None else _inputs(ANCHOR_KIND, value)
