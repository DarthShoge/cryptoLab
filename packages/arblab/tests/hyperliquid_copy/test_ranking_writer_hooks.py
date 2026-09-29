import os

import pyarrow.parquet as pq
import pytest

from arblab.hyperliquid_copy.disk_score_query import ScoreQuery
from arblab.hyperliquid_copy.disk_cohort_query import CohortQuery
from arblab.hyperliquid_copy.disk_metric_rows import write_metric_rows
from arblab.hyperliquid_copy.lab_config import METRICS, day
from .test_disk_score_query import resources
from .test_disk_metric_rows import output, row
from .test_lab_ranking import settings


@pytest.mark.parametrize("kind", ["scores", "rankings"])
@pytest.mark.parametrize("mode", ["capture", "failure", "invalid"])
def test_creation_hook_runs_before_parquet_writer(
    resources, tmp_path, monkeypatch, kind, mode
):
    config = settings()
    token = write_metric_rows(resources, [row()], METRICS)
    source = output(resources, token)
    if kind == "rankings":
        scores = tmp_path / "source-scores.parquet"
        with ScoreQuery(source, tmp_path, config) as query:
            query.write_scores(scores)
        query = CohortQuery(scores, tmp_path, config, day(config.start), "BTC")
    else:
        query = ScoreQuery(source, tmp_path, config)
    target = tmp_path / "output.parquet"
    captured = []

    def capture(fd):
        info = os.fstat(fd)
        assert info.st_size == 0
        assert os.pread(fd, 1, 0) == b""
        assert (info.st_dev, info.st_ino) == (
            target.stat().st_dev,
            target.stat().st_ino,
        )
        if mode == "failure":
            raise RuntimeError("capture rejected")
        captured.append(os.dup(fd))

    original = pq.ParquetWriter

    def checked_writer(*args, **kwargs):
        assert captured, "writer began before creation-time ownership capture"
        return original(*args, **kwargs)

    monkeypatch.setattr(pq, "ParquetWriter", checked_writer)
    try:
        with query:
            write = query.write_scores if kind == "scores" else query.write_rankings
            if mode == "invalid":
                with pytest.raises(ValueError):
                    write(target, on_created=True)
                assert not target.exists()
            elif mode == "failure":
                with pytest.raises(RuntimeError, match="capture rejected"):
                    write(target, on_created=capture)
                assert target.stat().st_size == 0
            else:
                write(target, on_created=capture)
                assert pq.read_metadata(target).num_rows == 1
                assert len(captured) == 1
                assert os.fstat(captured[0]).st_ino == target.stat().st_ino
    finally:
        for fd in captured:
            os.close(fd)
