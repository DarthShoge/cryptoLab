"""One causal query window per decision over reusable checkpoint history.

This adapter consumes an already validated frozen lineage. It does not qualify
archive coverage, acquire data, or lift the initial checkpoint build limits.
"""

from datetime import datetime, timedelta

from .contracts import utc
from .lab_config import day
from .lab_config_proxy import LabConfigProxyScheduled
from .lab_config_v2 import ExplicitUniverse


def history_days(config):
    follower, universe = config.follower, config.market_universe
    return max(
        config.trader.lookback_days,
        follower.scale_lookback_days
        if follower.aggregation == "conviction_trimmed"
        else 0,
        0
        if isinstance(universe, ExplicitUniverse)
        else universe.lookback_days + universe.publication_lag_days,
    )


class ScheduledActivity:
    def __init__(self, store, identity, config, *, temp_root, **limits):
        if not isinstance(config, LabConfigProxyScheduled):
            raise ValueError("Scheduled activity requires scheduled config")
        self.store, self.identity = store, identity
        self.cutoff = utc(datetime.fromisoformat(store.metadata(identity)["cutoff"]))
        self.start, self.end = day(config.start), day(config.end)
        self.lookback = timedelta(days=history_days(config))
        if self.cutoff > self.start - self.lookback:
            raise ValueError("Checkpoint lacks required decision warmup")
        self.temp_root, self.limits = temp_root, limits
        self.reader = None
        self.last_decision = None
        self.closed = False

    def prepare(self, at):
        if self.closed:
            raise ValueError("Scheduled activity is closed")
        at = utc(at)
        if (
            not self.start <= at < self.end
            or at.minute
            or at.second
            or at.microsecond
            or self.last_decision is not None
            and at < self.last_decision
        ):
            raise ValueError("Invalid or reversed scheduled decision")
        if at == self.last_decision and self.reader is not None:
            return
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        lower = at - self.lookback
        identity, cutoff = self.identity, self.cutoff
        if lower > self.cutoff:
            identity = self.store.advance(
                self.identity, lower, temp_root=self.temp_root, **self.limits
            )
            cutoff = lower
        reader = self.store.open(
            identity,
            lower,
            at + timedelta(hours=1),
            temp_root=self.temp_root,
            **self.limits,
        )
        self.identity, self.cutoff, self.reader = identity, cutoff, reader
        self.last_decision = at

    def _prepared(self):
        if self.closed or self.reader is None:
            raise ValueError("Scheduled activity must be prepared and not closed")
        return self.reader

    def observed(self, *args, **kwargs):
        return self._prepared().observed(*args, **kwargs)

    def position(self, *args, **kwargs):
        return self._prepared().position(*args, **kwargs)

    def rank(self, *args, **kwargs):
        return self._prepared().rank(*args, **kwargs)

    def volume(self, *args, **kwargs):
        return self._prepared().volume(*args, **kwargs)

    def hourly_exposure(self, *args, **kwargs):
        return self._prepared().hourly_exposure(*args, **kwargs)

    def close(self):
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        self.closed = True

    def __enter__(self):
        if self.closed:
            raise ValueError("Scheduled activity is closed")
        return self

    def __exit__(self, *_):
        self.close()
