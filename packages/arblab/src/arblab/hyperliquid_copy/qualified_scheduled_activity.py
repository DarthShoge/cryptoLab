"""Causal scheduled composition over qualified files and an exclusive cache loan.

No full-prefix activity/checkpoint reader is opened. Registered interiors require
an explicit source-seed policy; default coverage must match the source exactly.
Rankings use reusable chronological features; automatic retirement is separate.
"""

from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

from .archive_cache import _safe
from .annual_execution_policy import execution_policy, feature_execution_policy_name
from .annual_ranking_consumption import acknowledge_consumption
from .annual_ranking_retirement import (
    prepare_operation,
    execute_operation,
    recover_operation,
)
from .feature_metric_producer import _engine as metric_engine
from .feature_history import FeatureHistory, _engine as history_engine
from .feature_history_policy import checked_feature_policy
from .ranking_staging_policy import validate_policy as checked_staging_policy
from .ranking_staging_binding import _engine as staging_engine
from .rolling_feature_history import RollingFeatureHistory, _engine as rolling_engine
from .rolling_history_route import prepare_history_route, _engine as route_engine
from .contracts import address, utc
from .derived_publication import _encode
from .disk_cohort_scoring import _engine as score_engine
from .download import file_hash
from .lab_config import day
from .lab_config_proxy import LabConfigProxyScheduled
from .prefix_qualification import _previous, _engine as qualification_engine
from .qualified_hourly_exposure import hourly_exposure, _engine as exposure_engine
from .qualified_market_queries import (
    observed_markets,
    market_volume,
    _engine as market_engine,
)
from .qualified_native_positions import native_positions, _engine as position_engine
from .scheduled_activity import history_days
from .qualified_source_session import QualifiedSourceSession, _identity
from .saved_feature_rankings import SavedFeatureRankings, _engine as receipt_engine
from .saved_ranking_lookup import find_receipt_inputs
from .contracts import semantic_hash


def _engine():
    root = Path(__file__).parent
    return dict(
        code={
            name: file_hash(root / name)
            for name in (
                "qualified_scheduled_activity.py",
                "feature_history_policy.py",
                "ranking_staging_policy.py",
                "bound_scoring_context.py",
                "scheduled_activity.py",
                "lab_config.py",
                "lab_config_proxy.py",
                "lab_config_v2.py",
            )
        },
        metrics=metric_engine(),
        history=history_engine(),
        rolling=rolling_engine(),
        route=route_engine(),
        scoring=score_engine(),
        positions=position_engine(),
        markets=market_engine(),
        exposure=exposure_engine(),
        qualification=qualification_engine(),
        receipts=receipt_engine(),
        staging=staging_engine(),
    )


class QualifiedScheduledActivity:
    def __init__(
        self,
        resources,
        pin,
        config,
        *,
        coverage_start,
        coverage_end,
        semantics,
        history_policy=None,
        feature_history_policy=None,
        ranking_staging_policy=None,
        execution_policy_name=None,
        validation=None,
    ):
        resources.lease.check()
        checked_feature_policy(feature_history_policy)
        checked_staging_policy(ranking_staging_policy)
        annual_policy = execution_policy(execution_policy_name)
        if annual_policy is not None and (
            history_policy != "qualified_source_seed_v1"
            or feature_history_policy != "rolling_feature_anchor_v1"
            or ranking_staging_policy != "bounded_ranking_staging_v1"
        ):
            raise ValueError("Annual execution policy prerequisites are missing")
        if validation is not None and not callable(validation):
            raise ValueError("Invalid scheduled context validation")
        self._validation = validation
        if not isinstance(config, LabConfigProxyScheduled):
            raise ValueError("Qualified reader requires scheduled configuration")
        if semantics not in ("gross_excludes_fee", "net_includes_fee"):
            raise ValueError("Unresolved scheduled metric semantics")
        if type(pin) is not dict or set(pin) != {"path", "sha256"}:
            raise ValueError("Pinned qualification report required")
        report = _previous(pin, qualification_engine())
        origin, source_end = day(report["source_start"]), day(report["source_end"])
        coverage_begin, finish = day(coverage_start), day(coverage_end)
        if history_policy is None:
            if (coverage_start, coverage_end) != (
                report["source_start"],
                report["source_end"],
            ):
                raise ValueError(
                    "Declared coverage must match qualified source origin/end"
                )
        elif history_policy != "qualified_source_seed_v1" or not origin + timedelta(
            days=1
        ) <= coverage_begin < finish <= source_end - timedelta(days=1):
            raise ValueError("Invalid qualified source-seed policy/padded coverage")
        start, end = day(config.start), day(config.end)
        if (
            not coverage_begin
            <= start - timedelta(days=history_days(config))
            < start
            < end
            <= finish
        ):
            raise ValueError("Missing scheduled source coverage/warmup")
        self.resources, self.config = resources, config
        self._caller_pin, self.pin = pin, dict(pin)
        self.origin, self.finish = origin, finish
        self.coverage_start, self.coverage_end = coverage_start, coverage_end
        self.history_policy = history_policy
        self.feature_history_policy = feature_history_policy
        self.ranking_staging_policy = ranking_staging_policy
        self.execution_policy = None if annual_policy is None else annual_policy.name
        self.feature_execution_policy = (
            None
            if annual_policy is None
            else feature_execution_policy_name(annual_policy.name, config.rebalance)
        )
        self.start, self.end, self.semantics = start, end, semantics
        self.coins = tuple(report["coins"])
        self.closed, self._busy, self.last_decision = False, False, None
        self._features = None
        self._rankings = None
        self._ranking_source = None
        self._ranking_receipts = {}
        self._failed_consumption = False
        self._failed_recovery = None
        self._frozen_inputs = _encode(self._inputs())
        self._frozen = self._context()
        self._verify()

    def _inputs(self):
        return dict(
            pin=self.pin,
            config=self.config.to_dict(),
            origin=self.origin.isoformat(),
            finish=self.finish.isoformat(),
            coverage_start=self.coverage_start,
            coverage_end=self.coverage_end,
            history_policy=self.history_policy,
            feature_history_policy=self.feature_history_policy,
            ranking_staging_policy=self.ranking_staging_policy,
            execution_policy=self.execution_policy,
            feature_execution_policy=self.feature_execution_policy,
            start=self.start.isoformat(),
            end=self.end.isoformat(),
            semantics=self.semantics,
            coins=self.coins,
        )

    def _context(self):
        return _encode(dict(self._inputs(), engine=_engine()))

    def _verify(self):
        self.resources.lease.check()
        if self.closed:
            raise ValueError("Qualified scheduled activity is closed")
        path = Path(self.pin["path"])
        _safe(path)
        if (
            file_hash(path) != self.pin["sha256"]
            or self._caller_pin != self.pin
            or self._context() != self._frozen
        ):
            raise ValueError("Qualified scheduled source/config/engine changed")
        if self._validation is not None:
            self._validation()
        if (
            self._caller_pin != self.pin
            or _encode(self._inputs()) != self._frozen_inputs
            or self.closed
        ):
            raise ValueError("Qualified scheduled context changed during validation")
        self.resources.lease.check()

    @contextmanager
    def _operation(self, *, prepared=True):
        if self._failed_consumption:
            raise ValueError(
                "Qualified scheduled activity requires explicit retirement recovery"
            )
        if self._busy:
            raise ValueError("Qualified scheduled activity is busy")
        self._verify()
        if prepared and self.last_decision is None:
            raise ValueError("Qualified scheduled activity must be prepared")
        cutoff = self.last_decision
        self._busy = True
        try:
            yield
        finally:
            try:
                self._verify()
                if self.last_decision != cutoff:
                    raise ValueError("Prepared decision changed during query")
            finally:
                self._busy = False

    def prepare(self, at):
        with self._operation(prepared=False):
            at = utc(at)
            if (
                not self.start <= at < self.end
                or at.minute
                or at.second
                or at.microsecond
                or self.last_decision is not None
                and at < self.last_decision
            ):
                raise ValueError("Invalid or reversed qualified scheduled decision")
        # The caller lends exclusive use of this facade/cache. Publish only
        # after the final context verification succeeds, never on failed prepare.
        self.last_decision = at

    def _decision(self, at):
        at = utc(at)
        if at != self.last_decision:
            raise ValueError("Query must match prepared decision")
        return at

    def observed(self, decision):
        with self._operation():
            return observed_markets(self.resources, self.pin, self._decision(decision))

    def positions(self, users, coin, decision):
        with self._operation():
            return native_positions(
                self.resources, self.pin, self._decision(decision), coin, users
            )

    def position(self, user, coin, decision):
        return self.positions([user], coin, decision)[address(user)]

    def volume(self, coin, start, end):
        with self._operation():
            start, end = utc(start), utc(end)
            if not self.origin <= start < end <= self.last_decision:
                raise ValueError("Volume outside prepared history")
            return market_volume(self.resources, self.pin, coin, start, end)

    def hourly_exposure(self, user, coin, start, end, *, max_price_age_seconds):
        with self._operation():
            start, end = utc(start), utc(end)
            if (
                not self.origin
                <= start
                < end
                <= min(self.finish, self.last_decision + timedelta(hours=1))
            ):
                raise ValueError("Hourly samples outside prepared history")
            return hourly_exposure(
                self.resources,
                self.pin,
                user,
                coin,
                start,
                end,
                max_price_age_seconds=max_price_age_seconds,
            )

    def rank(self, decision, config, scope, semantics, *, smoke=False):
        with self._operation():
            decision = self._decision(decision)
            if semantics != self.semantics or smoke is not True:
                raise ValueError("Scheduled metric semantics/smoke mismatch")
            expected = self.config.effective(config.coins, config.asset_weights)
            frozen = _encode(vars(config))
            if (
                frozen != _encode(vars(expected))
                or not set(config.coins) <= set(self.coins)
                or len(set(config.coins)) != len(config.coins)
                or scope is not None
                and scope not in config.coins
            ):
                raise ValueError("Effective ranking configuration/scope mismatch")
            if not config.coins:
                return []  # Explicitly empty active universe, not an empty artifact.
            if self._rankings is None:
                self._ranking_source = QualifiedSourceSession(self.pin)
                self._rankings = SavedFeatureRankings(
                    self.resources,
                    self._ranking_source,
                    staging_policy=self.ranking_staging_policy,
                    execution_policy_name=self.execution_policy,
                )
            result = self._rankings.find(decision, config, scope, semantics)
            if result is None:

                def validate_rank():
                    self._verify()
                    self._decision(decision)
                    if _encode(vars(config)) != frozen:
                        raise ValueError("Effective ranking configuration changed")

                start = decision - timedelta(days=config.lookback_days)
                if self.ranking_staging_policy is not None:
                    with self.resources._connect() as db:
                        if db.execute(
                            "SELECT 1 FROM allocations WHERE state='pending' LIMIT 1"
                        ).fetchone():
                            raise ValueError(
                                "Pending obligations block staged production"
                            )
                route = "features"
                if self.feature_history_policy is not None:
                    route = prepare_history_route(
                        self.resources,
                        self._ranking_source,
                        sorted(self.coins),
                        semantics,
                        start,
                        decision,
                        validation=validate_rank,
                        execution_policy_name=self.feature_execution_policy,
                    )
                if route == "raw":
                    if self.ranking_staging_policy is None:
                        receipt = self._rankings.capture_raw(
                            decision, config, scope, semantics
                        )
                    else:
                        receipt = self._rankings.capture_staged(
                            decision, config, scope, semantics, validation=validate_rank
                        )
                else:
                    if self._features is None:
                        history_type = (
                            FeatureHistory
                            if self.feature_history_policy is None
                            else RollingFeatureHistory
                        )
                        self._features = history_type(
                            self.resources,
                            self.pin,
                            sorted(self.coins),
                            semantics,
                            **(
                                {"execution_policy_name": self.feature_execution_policy}
                                if self.execution_policy is not None
                                else {}
                            ),
                        )
                    options = (
                        {}
                        if self.feature_history_policy is None
                        else dict(validation=validate_rank)
                    )
                    window = self._features.window(start, decision, **options)
                    validate_rank()
                    if self.ranking_staging_policy is None:
                        receipt = self._rankings.capture(
                            window.days, decision, config, scope, semantics
                        )
                    else:
                        receipt = self._rankings.capture_staged(
                            decision,
                            config,
                            scope,
                            semantics,
                            days=window.days,
                            validation=validate_rank,
                        )
                result = self._rankings.load(
                    receipt, decision, config, scope, semantics
                )
            if self.execution_policy is not None:
                receipt = find_receipt_inputs(
                    self.resources,
                    self._rankings._query(decision, config, scope, semantics),
                )
                if receipt is None:
                    raise ValueError("Annual saved ranking receipt is missing")
                self._ranking_receipts[result.publication.key] = receipt
            if _encode(vars(config)) != frozen:
                raise ValueError("Effective ranking configuration changed")
            self._decision(decision)
            identity = _identity(result.artifact.path)
            result.artifact.verify()
            self._ranking_source.verify()
            self._verify()
        # Keep evidence guarded through the operation's final facade checks.
        self._ranking_source.verify_identity()
        if _identity(result.artifact.path) != identity:
            raise ValueError("Scheduled ranking artifact identity changed")
        if _encode(vars(config)) != frozen:
            raise ValueError("Effective ranking configuration changed")
        self._decision(decision)
        self.resources.lease.check()
        return result

    def consume_ranking(self, result, snapshot, selected, *, config, semantics):
        if self.execution_policy is None:
            return None
        operation = receipt = None
        try:
            with self._operation():
                receipt = self._ranking_receipts.get(result.publication.key)
                if receipt is None:
                    raise ValueError("Annual ranking was not produced by this activity")
                config_sha256 = semantic_hash(vars(config))
                consumption = acknowledge_consumption(
                    result,
                    snapshot,
                    selected,
                    execution_policy=self.execution_policy,
                    source_pin=self.pin,
                    config_sha256=config_sha256,
                )
                operation = prepare_operation(self.resources, receipt, consumption)

                def verify_saved():
                    loaded = self._rankings.load(
                        receipt,
                        result.bound_decision,
                        config,
                        result.bound_scope,
                        semantics,
                    )
                    if (
                        loaded.candidate_count != result.candidate_count
                        or loaded.eligible_count != result.eligible_count
                        or loaded._selected_json != result._selected_json
                    ):
                        raise ValueError("Annual saved ranking reload changed")

                outcome = execute_operation(
                    self.resources, operation, verify_saved=verify_saved
                )
                del self._ranking_receipts[result.publication.key]
                return outcome
        except BaseException:
            self._failed_consumption = True
            if operation is not None:
                self._failed_recovery = (
                    operation,
                    receipt,
                    result,
                    config,
                    semantics,
                )
            raise

    def recover_consumption(self):
        """Explicitly resume this activity's exact journaled retirement."""
        if not self._failed_consumption or self._failed_recovery is None or self._busy:
            raise ValueError("No recoverable annual ranking consumption")
        self._verify()
        operation, receipt, result, config, semantics = self._failed_recovery

        def verify_saved():
            loaded = self._rankings.load(
                receipt,
                result.bound_decision,
                config,
                result.bound_scope,
                semantics,
            )
            if (
                loaded.candidate_count != result.candidate_count
                or loaded.eligible_count != result.eligible_count
                or loaded._selected_json != result._selected_json
            ):
                raise ValueError("Annual saved ranking reload changed")

        outcome = recover_operation(
            self.resources, operation, verify_saved=verify_saved
        )
        self._ranking_receipts.pop(result.publication.key, None)
        self._failed_recovery = None
        self._failed_consumption = False
        self._verify()
        return outcome

    def close(self):
        if self._busy:
            raise ValueError("Cannot close busy qualified scheduled activity")
        self.closed = True

    def __enter__(self):
        self._verify()
        return self

    def __exit__(self, *_):
        self.close()
