import io
import json

import lz4.frame
import pytest

from .test_archive import USER, raw_fill


class S3:
    def __init__(self, missing=None):
        self.missing = missing
        self.reads = 0

    def head_object(self, *, Key, **kwargs):
        if Key.endswith(f"/{self.missing}.lz4"):
            raise ValueError("missing hour")
        return {"ContentLength": 123}

    def get_object(self, **kwargs):
        self.reads += 1
        raw = json.dumps([USER, raw_fill()]).encode() + b"\n"
        return {"Body": io.BytesIO(lz4.frame.compress(raw))}


def test_atomic_download_complete_and_idempotent(tmp_path):
    from arblab.hyperliquid_copy.download import pull_fill_day
    s3 = S3()
    result = pull_fill_day(s3, tmp_path, "2025-07-26", coins=("BTC",), accepted_cost=True)
    assert result.exists()
    assert s3.reads == 24
    assert pull_fill_day(s3, tmp_path, "2025-07-26", coins=("BTC",), accepted_cost=True) == result
    assert s3.reads == 24


def test_no_spend_without_consent_and_missing_hour_leaves_no_partition(tmp_path):
    from arblab.hyperliquid_copy.download import pull_fill_day
    with pytest.raises(ValueError, match="cost"):
        pull_fill_day(S3(), tmp_path, "2025-07-26")
    with pytest.raises(ValueError, match="missing"):
        pull_fill_day(S3(23), tmp_path, "2025-07-26", accepted_cost=True)
    assert not list(tmp_path.rglob("fills.parquet"))
