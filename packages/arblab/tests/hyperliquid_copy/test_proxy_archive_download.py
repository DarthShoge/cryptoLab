import io
import json
import subprocess
import sys
from pathlib import Path

import pytest

from arblab.hyperliquid_copy.proxy_archive_download import download_archive


class Source:
    def __init__(self, body=b"abc", declared=3):
        self.body, self.declared, self.gets = body, declared, []

    def head_object(self, **kwargs):
        assert kwargs["RequestPayer"] == "requester"
        return {"ContentLength": self.declared, "ETag": '"fixed"'}

    def get_object(self, **kwargs):
        assert kwargs["IfMatch"] == '"fixed"'
        assert kwargs["RequestPayer"] == "requester"
        self.gets.append(kwargs["Key"])
        return {
            "ContentLength": self.declared,
            "ETag": '"fixed"',
            "Body": io.BytesIO(self.body),
        }


def test_size_budget_checked_before_any_paid_get(tmp_path):
    source = Source()
    with pytest.raises(ValueError, match="budget"):
        download_archive(source, "2026-08-01", "2026-08-02", tmp_path, max_bytes=71)
    assert not source.gets


def test_frozen_objects_downloaded_once_with_audit_and_hashes(tmp_path):
    source = Source()
    path = download_archive(source, "2026-08-01", "2026-08-02", tmp_path, max_bytes=72)
    manifest = json.loads(path.read_text())
    assert manifest["complete"] and manifest["reserved_bytes"] == 72
    assert len(source.gets) == len(set(source.gets)) == 24
    assert all(
        (path.parent / o["file"]).read_bytes() == b"abc" for o in manifest["objects"]
    )
    assert all(len(o["sha256"]) == 64 for o in manifest["objects"])


@pytest.mark.parametrize("body", [b"ab", b"abcd"])
def test_truncated_or_oversized_transfer_never_publishes_success(tmp_path, body):
    source = Source(body)
    with pytest.raises(ValueError, match="length"):
        download_archive(source, "2026-08-01", "2026-08-02", tmp_path, max_bytes=72)
    manifest = json.loads(next(tmp_path.glob("*/manifest.json")).read_text())
    assert not manifest["complete"]
    assert manifest["reserved_bytes"] == 3
    assert len(source.gets) == 1
    assert not list(tmp_path.glob("*/*.lz4"))


def interrupted(tmp_path):
    source = Source()

    def stop(item):
        if item.get("downloaded") == 1:
            raise RuntimeError("Stopped after durable object")

    with pytest.raises(RuntimeError):
        download_archive(
            source, "2026-08-01", "2026-08-02", tmp_path, max_bytes=72, progress=stop
        )
    return source, next(tmp_path.glob("*/manifest.json"))


def test_resume_skips_verified_objects_and_preserves_total_budget(tmp_path):
    from arblab.hyperliquid_copy.proxy_archive_download import resume_archive

    source, path = interrupted(tmp_path)
    assert resume_archive(source, path) == path
    assert len(source.gets) == len(set(source.gets)) == 24
    data = json.loads(path.read_text())
    assert data["complete"] and data["reserved_bytes"] == data["max_bytes"] == 72
    assert resume_archive(source, path) == path
    assert len(source.gets) == 24


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "escape",
        "duplicate",
        "budget",
        "reservation",
        "key",
        "requested",
        "unexpected",
    ],
)
def test_resume_rejects_unsafe_state_before_network(tmp_path, fault):
    from arblab.hyperliquid_copy.proxy_archive_download import resume_archive

    source, path = interrupted(tmp_path)
    data = json.loads(path.read_text())
    first, second = data["objects"][:2]
    if fault == "hash":
        (path.parent / first["file"]).write_bytes(b"bad")
    elif fault == "escape":
        first["file"] = "../outside.lz4"
    elif fault == "duplicate":
        second["file"] = first["file"]
    elif fault == "budget":
        data["max_bytes"] = 71
    elif fault == "reservation":
        data["reserved_bytes"] = 0
    elif fault == "key":
        second["key"] = first["key"]
    elif fault == "requested":
        second["status"] = "requested"
        data["reserved_bytes"] += second["bytes"]
    else:
        (path.parent / second["file"]).write_bytes(b"abc")
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        resume_archive(source, path)
    assert len(source.gets) == 1


def test_resume_serializes_with_active_original_downloader(tmp_path):
    from arblab.hyperliquid_copy.proxy_archive_download import resume_archive

    source = Source()

    def check_lock(item):
        if "manifest" in item:
            with pytest.raises(ValueError, match="already running"):
                resume_archive(source, item["manifest"])
            assert not source.gets

    download_archive(
        source, "2026-08-01", "2026-08-02", tmp_path, max_bytes=72, progress=check_lock
    )
    assert len(source.gets) == 24


def test_resume_publishes_completion_after_last_object_without_another_get(tmp_path):
    from arblab.hyperliquid_copy.proxy_archive_download import resume_archive

    source = Source()

    def stop(item):
        if item.get("downloaded") == 24:
            raise RuntimeError("Before completion flag")

    with pytest.raises(RuntimeError):
        download_archive(
            source, "2026-08-01", "2026-08-02", tmp_path, max_bytes=72, progress=stop
        )
    path = next(tmp_path.glob("*/manifest.json"))
    assert not json.loads(path.read_text())["complete"]
    resume_archive(source, path)
    assert json.loads(path.read_text())["complete"]
    assert len(source.gets) == 24


@pytest.mark.parametrize(
    "link_target", ["manifest.json", ".acquisition.lock", "fills_0000.lz4"]
)
def test_resume_rejects_symlinks_before_network(tmp_path, link_target):
    from arblab.hyperliquid_copy.proxy_archive_download import resume_archive

    source, path = interrupted(tmp_path)
    original = path.parent / link_target
    moved = path.parent / (link_target + ".saved")
    original.rename(moved)
    original.symlink_to(moved)
    with pytest.raises(ValueError, match="[Ss]ymlink|[Uu]nsafe"):
        resume_archive(source, path)
    assert len(source.gets) == 1


def test_cli_exposes_resume_but_requires_explicit_approval(tmp_path):
    tool = (
        Path(__file__).resolve().parents[4]
        / "tools/download_hyperliquid_proxy_archive.py"
    )
    help_result = subprocess.run(
        [sys.executable, str(tool), "--help"], capture_output=True, text=True
    )
    assert help_result.returncode == 0
    assert "--resume-manifest" in help_result.stdout
    result = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--resume-manifest",
            str(tmp_path / "manifest.json"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Explicit approval required" in result.stderr
