"""Constant-space order/pairing validation across feature batches and shards."""

from .feature_records import KEY_FIELDS


class ObservationOrder:
    def __init__(self):
        self.previous = self.pending = None

    def add(self, row):
        key = (row["user"], *(row[field] for field in KEY_FIELDS))
        if self.pending is not None:
            if row["kind"] != "fill" or (key, row["coin"]) != self.pending:
                raise ValueError("Episode must be followed by its same-key fill")
            self.pending, self.previous = None, key
        else:
            if self.previous is not None and key <= self.previous:
                raise ValueError("Duplicate or unordered feature native key")
            if row["kind"] == "episode":
                self.pending = (key, row["coin"])
            else:
                self.previous = key

    def finish(self):
        if self.pending is not None:
            raise ValueError("Dangling completed episode without closing fill")
