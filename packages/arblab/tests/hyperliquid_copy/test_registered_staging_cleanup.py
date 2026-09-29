import json

import pytest

from arblab.hyperliquid_copy.lab_config import day
from arblab.hyperliquid_copy.ranking_staging_policy import POLICY
from .test_qualified_registration import registered_source, resources
from .test_qualified_registered_activity import load
from .test_scheduled_staged_rankings import arm_final_cleanup_engine


@pytest.mark.parametrize("fault", ["disk", "metadata", "facade"])
def test_final_registered_cleanup_mutation_preserves_temporaries(
    registered_source, resources, monkeypatch, fault
):
    from arblab.hyperliquid_copy import qualified_registered_activity as module

    manifest, config = registered_source
    manifest.metadata["ranking_staging_policy"] = POLICY
    path = manifest.directory / "manifest.json"
    path.write_text(json.dumps(manifest.metadata))
    resources.lease.__exit__(None, None, None)
    owned = load(manifest, config)
    at = day(config.start)
    owned.prepare(at)

    def mutate():
        if fault == "disk":
            metadata = json.loads(path.read_text())
            metadata.pop("ranking_staging_policy")
            path.write_text(json.dumps(metadata))
        elif fault == "metadata":
            manifest.metadata.pop("ranking_staging_policy")
        else:
            owned._activity.ranking_staging_policy = None

    state = arm_final_cleanup_engine(monkeypatch, module, mutate)
    try:
        with pytest.raises(ValueError):
            owned.rank(
                at,
                config.effective(["BTC"], {"BTC": 1}),
                "BTC",
                "gross_excludes_fee",
                smoke=True,
            )
        assert state["changed"] and state["paths"]
        assert all(path.exists() for path in state["paths"])
        assert len(list((resources.root / "staging").glob("*.parquet"))) == 2
    finally:
        with pytest.raises(ValueError):
            owned.close()
