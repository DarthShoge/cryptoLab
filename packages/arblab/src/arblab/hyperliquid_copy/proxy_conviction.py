"""Bounded hourly native-notional normalization; future samples never set a scale."""

from datetime import timedelta
from .contracts import utc


class ProxyConviction:
    def __init__(self, activity, config, end):
        self.activity, self.config, self.end = activity, config, end
        self.samples = {}
        self.last_decision = None

    def advance(self, required, at):
        at = utc(at)
        if (
            at >= utc(self.end)
            or at.minute
            or at.second
            or at.microsecond
            or self.last_decision is not None
            and at < self.last_decision
        ):
            raise ValueError("Invalid or reversed conviction decision")
        days = self.config.follower.scale_lookback_days
        begin = at - timedelta(days=days)
        finish = at + timedelta(hours=1)
        sample_count = days * 24 + 1
        if sample_count * len(required) > 1_000_000:
            raise ValueError("Native conviction history ceiling exceeded")
        self.samples = {
            key: {time: value for time, value in samples.items() if begin <= time <= at}
            for key, samples in self.samples.items()
            if key in required
        }
        for user, coin in sorted(required):
            samples = self.samples.setdefault((user, coin), {})
            fresh_begin = max(
                begin,
                max(samples, default=begin - timedelta(hours=1)) + timedelta(hours=1),
            )
            if fresh_begin < finish:
                samples.update(
                    self.activity.hourly_exposure(
                        user,
                        coin,
                        fresh_begin,
                        finish,
                        max_price_age_seconds=self.config.proxy.max_mark_age_seconds,
                    )
                )
        self.last_decision = at

    def value(self, key, at):
        samples = self.samples[key]
        current = samples.get(at)
        begin = at - timedelta(days=self.config.follower.scale_lookback_days)
        past = sorted(
            abs(value)
            for time, value in samples.items()
            if begin <= time < at and value is not None
        )
        if current is None or not past:
            return None
        index = (len(past) - 1) * self.config.follower.scale_quantile
        low = int(index)
        high = min(low + 1, len(past) - 1)
        scale = past[low] + (past[high] - past[low]) * (index - low)
        return max(-1.0, min(1.0, current / scale)) if scale > 0 else None
