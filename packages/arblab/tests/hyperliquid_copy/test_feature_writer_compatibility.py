"""Only the exact storage-only fix may reuse the pre-fix feature cache identity."""

from arblab.hyperliquid_copy import feature_publication

LEGACY = {
    "feature_writer.py": "059f867cd804f9adde7b4d87f541d6b10ff61e6bab00fecdd69bc0d1a89260b8",
    "feature_publication.py": "3e02738fc88a4cf65088cae54ad7c4b72c338af06de4703c388304956c988dca",
}


def test_storage_only_fix_retains_legacy_feature_identity():
    engine = feature_publication.feature_engine()
    assert {name: engine["code"][name] for name in LEGACY} == LEGACY


def test_unknown_writer_or_budget_change_invalidates_compatibility(monkeypatch):
    from arblab.hyperliquid_copy import feature_writer_compatibility as module

    original = module.file_hash
    for name in (
        "feature_writer.py",
        "feature_writer_budget.py",
        "feature_publication.py",
    ):
        with monkeypatch.context() as patch:
            patch.setattr(
                module,
                "file_hash",
                lambda p: "f" * 64 if p.name == name else original(p),
            )
            code = feature_publication.feature_engine()["code"]
            assert any(code[key] != value for key, value in LEGACY.items())
