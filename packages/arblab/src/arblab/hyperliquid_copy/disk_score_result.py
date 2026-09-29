"""Immutable bounded cohort metadata and verified ranking-evidence batches."""

from dataclasses import dataclass
from datetime import datetime
import json
from math import ceil

import pyarrow.parquet as pq

from .derived_publication import Publication
from .disk_metric_rows import MAX_BYTES, MAX_GROUPS, MAX_ROWS
from .ranking_artifact import RankingFile, RANKING_SCHEMA


def ranking_batches(artifact):
    artifact.verify()
    try:
        with pq.ParquetFile(artifact.path) as source:
            if (
                source.schema_arrow != RANKING_SCHEMA
                or source.metadata.num_rows != artifact.rows
                or artifact.rows > MAX_ROWS
                or artifact.bytes > MAX_BYTES
                or source.metadata.num_row_groups > MAX_GROUPS
            ):
                raise ValueError("Invalid ranking artifact schema/metadata")
            for batch in source.iter_batches(batch_size=4096):
                if batch.nbytes > 64 * 1024**2:
                    raise ValueError("Ranking decoded batch limit exceeded")
                rows = batch.to_pylist()
                for row in rows:
                    row["percentiles"] = {
                        k: v for k, v in row["percentiles"].items() if v is not None
                    }
                yield rows
    finally:
        artifact.verify()


@dataclass(frozen=True)
class ScoredCohort:
    publication: Publication
    artifact: RankingFile
    candidate_count: int
    eligible_count: int
    requested_count: int
    _selected_json: tuple[str, ...]

    @property
    def selected(self):
        result = []
        for raw in self._selected_json:
            row = json.loads(raw)
            row["decision_time"] = datetime.fromisoformat(row["decision_time"])
            result.append(row)
        return tuple(result)

    @property
    def cutoff_address(self):
        return (
            json.loads(self._selected_json[-1])["user"] if self._selected_json else None
        )

    def iter_batches(self):
        return ranking_batches(self.artifact)


def read_cohort(root, publication, selection):
    if len(publication.artifacts) != 1:
        raise ValueError("Expected one ranking artifact")
    pin = publication.artifacts[0]
    path = root / pin.path
    artifact = RankingFile(path, pq.read_metadata(path).num_rows, pin.bytes, pin.sha256)
    count = eligible = 0
    selected = []
    for batch in ranking_batches(artifact):
        count += len(batch)
        for row in batch:
            eligible += int(row["eligible"])
            if row["selected"]:
                if len(selected) >= selection["maximum"]:
                    raise ValueError("Selected cohort limit exceeded")
                selected.append(
                    json.dumps(
                        {
                            k: (v.isoformat() if k == "decision_time" else v)
                            for k, v in row.items()
                            if k not in ("market_decision_time", "decision_trigger")
                        },
                        sort_keys=True,
                        allow_nan=False,
                    )
                )
    requested = (
        ceil(selection["top_fraction"] * eligible)
        if selection["selection"] == "fraction"
        else selection["top_n"]
    )
    size = (
        min(eligible, selection["maximum"], max(selection["minimum"], requested))
        if eligible >= selection["minimum"]
        else 0
    )
    if len(selected) != size:
        raise ValueError("Published cohort size mismatch")
    return ScoredCohort(
        publication, artifact, count, eligible, requested, tuple(selected)
    )
