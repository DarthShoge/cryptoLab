"""Point-in-time position reconstruction without zero-position assumptions."""
from bisect import bisect_left, bisect_right

from .contracts import utc


class PositionReplay:
    def __init__(self, fills):
        self.history = {}
        self.times = {}
        for fill in sorted(fills, key=lambda f: f.order_key):
            key = fill.user, fill.coin
            self.history.setdefault(key, []).append(fill.post_position)
            self.times.setdefault(key, []).append(fill.exchange_time)

    def snapshot(self, at, *, inclusive=True):
        at = utc(at)
        bisect = bisect_right if inclusive else bisect_left
        result = {}
        for key, times in self.times.items():
            index = bisect(times, at) - 1
            if index >= 0:
                result[key] = self.history[key][index]
        return result

    def position(self, key, at, *, inclusive=True):
        bisect = bisect_right if inclusive else bisect_left
        index = bisect(self.times.get(key, []), utc(at))-1
        return self.history[key][index] if index >= 0 else None
