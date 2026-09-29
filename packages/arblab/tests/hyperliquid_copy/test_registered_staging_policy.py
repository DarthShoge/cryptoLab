import json
from types import SimpleNamespace

import pytest

from arblab.hyperliquid_copy.ranking_staging_policy import POLICY


@pytest.mark.parametrize("value", ["unknown", False, {}])
def test_registration_rejects_unknown_staging_before_source_io(tmp_path, value):
    from arblab.hyperliquid_copy.annual_registration import register_annual_dataset

    with pytest.raises(ValueError, match="ranking staging policy"):
        register_annual_dataset(
            None,
            None,
            None,
            tmp_path / "target",
            config=None,
            name="test",
            ranking_staging_policy=value,
        )
    assert not list(tmp_path.iterdir())


def test_staging_requires_explicit_shared_cache_before_source_io(tmp_path):
    from arblab.hyperliquid_copy.annual_registration import register_annual_dataset

    with pytest.raises(ValueError, match="shared cache"):
        register_annual_dataset(
            None,
            None,
            None,
            tmp_path / "target",
            config=None,
            name="test",
            ranking_staging_policy=POLICY,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", ["unknown", POLICY])
def test_manifest_rejects_unknown_or_unqualified_staging(tmp_path, value):
    from arblab.hyperliquid_copy.proxy_dataset import ProxyDatasetManifest, SCHEMA

    (tmp_path / "manifest.json").write_text(
        json.dumps(dict(schema=SCHEMA, synthetic=True, ranking_staging_policy=value))
    )
    with pytest.raises(ValueError, match="[Ss]taging|staging"):
        ProxyDatasetManifest(tmp_path)


def test_qualified_verifier_rejects_unknown_staging_before_source_io(tmp_path):
    from arblab.hyperliquid_copy.qualified_registration import _verify

    manifest = SimpleNamespace(
        directory=tmp_path, metadata={"ranking_staging_policy": "unknown"}
    )
    with pytest.raises(ValueError, match="ranking staging policy"):
        _verify(manifest)
