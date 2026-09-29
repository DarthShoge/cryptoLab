from datetime import timedelta
from pathlib import Path
import json
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from .test_candidate_day import resources
from .test_feature_day_builder import DAY, SEMANTICS, publication_count
from .test_feature_history import history
from .test_qualified_day import qualified


def test_publishes_retained_tail_with_complete_checkpoint_evidence(
    resources, qualified
):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor

    days = history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days
    before = resources.audit()
    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    assert anchor.first == DAY + timedelta(days=1)
    assert anchor.cutoff == DAY + timedelta(days=3)
    assert tuple(d.inputs for d in anchor.days) == tuple(d.inputs for d in days[1:])
    assert list(anchor.days[-1].checkpoints()) == list(days[-1].checkpoints())
    assert list(anchor.days[-1].checkpoints())
    assert publication_count(resources) == 3
    after = resources.audit()
    assert after["reserved_bytes"] == 0
    assert (
        after["retained_bytes"] - before["retained_bytes"]
        == anchor.publication.artifacts[0].bytes
    )
    inputs = anchor.inputs
    inputs["coins"].append("ETH")
    assert anchor.inputs["coins"] == ["BTC"]
    anchor.verify()


@pytest.fixture
def days(resources, qualified):
    return history(resources, qualified).window(DAY, DAY + timedelta(days=3)).days


def test_fresh_lease_reopen_and_reuse_skip_preboundary_days(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_day_builder
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.feature_publication import FeatureDay
    from arblab.hyperliquid_copy.feature_resume_anchor import (
        FeatureAnchor,
        publish_feature_anchor,
    )
    from arblab.hyperliquid_copy.ordered_wallet_partitions import (
        OrderedWalletPartitions,
    )

    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    expected = [(list(d.observations()), list(d.checkpoints())) for d in anchor.days]
    inputs, before = anchor.inputs, resources.audit()
    resources.lease.__exit__(None)
    original = FeatureDay.verify

    def verify(day):
        assert day.day != DAY, "anchor must not verify a pre-boundary day"
        return original(day)

    def forbidden(*args, **kwargs):
        pytest.fail("anchor reopen must not derive or sort")

    monkeypatch.setattr(FeatureDay, "verify", verify)
    monkeypatch.setattr(feature_day_builder, "build_feature_day", forbidden)
    monkeypatch.setattr(OrderedWalletPartitions, "plan", forbidden)
    with CacheLease(resources.root) as lease:
        reopened = CacheResources(lease, resources.identity)
        result = FeatureAnchor(reopened, qualified, inputs)
        assert [
            (list(d.observations()), list(d.checkpoints())) for d in result.days
        ] == expected
        reused = publish_feature_anchor(
            reopened, qualified, result.days, ["BTC"], SEMANTICS
        )
        assert reused.publication == anchor.publication
        assert reopened.audit() == before


@pytest.mark.parametrize(
    "fault",
    ["empty", "large", "gap", "reverse", "wrong_type", "none", "scope", "semantics"],
)
def test_invalid_intervals_do_not_allocate(resources, qualified, days, fault):
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor

    selected, coins, semantics = days, ["BTC"], SEMANTICS
    if fault == "empty":
        selected = []
    elif fault == "large":
        selected = [days[0]] * 733
    elif fault == "gap":
        selected = (days[0], days[2])
    elif fault == "reverse":
        selected = days[::-1]
    elif fault == "wrong_type":
        selected = [object()]
    elif fault == "none":
        selected = None
    elif fault == "scope":
        coins = ["ETH"]
    else:
        semantics = "net_includes_fee"
    before = resources.audit()
    with pytest.raises(ValueError):
        publish_feature_anchor(resources, qualified, selected, coins, semantics)
    assert resources.audit() == before


@pytest.mark.parametrize("target", ["anchor", "day", "report"])
def test_existing_anchor_rejects_dependency_corruption(
    resources, qualified, days, target
):
    from arblab.hyperliquid_copy.feature_resume_anchor import (
        FeatureAnchor,
        publish_feature_anchor,
    )

    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    if target == "anchor":
        path = resources.root / anchor.publication.artifacts[0].path
    elif target == "day":
        path = resources.root / days[-1].publication.artifacts[0].path
    else:
        path = Path(qualified["path"])
    with path.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError):
        FeatureAnchor(resources, qualified, anchor.inputs)
    with pytest.raises(ValueError):
        anchor.verify()


@pytest.mark.parametrize("target", ["anchor", "day", "report"])
def test_mutation_during_final_publication_lookup_is_rejected(
    resources, qualified, days, monkeypatch, target
):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy.feature_resume_anchor import (
        KIND,
        publish_feature_anchor,
    )

    anchor = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    path = (
        resources.root / anchor.publication.artifacts[0].path
        if target == "anchor"
        else resources.root / days[-1].publication.artifacts[0].path
        if target == "day"
        else Path(qualified["path"])
    )
    original = PublishedArtifacts.lookup

    def changed(self, kind, inputs):
        result = original(self, kind, inputs)
        if kind == KIND:
            with path.open("ab") as handle:
                handle.write(b"changed")
        return result

    monkeypatch.setattr(PublishedArtifacts, "lookup", changed)
    with pytest.raises(ValueError):
        anchor.verify()


def test_interruption_after_reservation_stays_charged(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy.feature_resume_anchor import (
        KIND,
        MAX_BYTES,
        publish_feature_anchor,
    )

    before = resources.audit()
    reserve = resources.reserve

    def interrupted(*args):
        reserve(*args)
        raise RuntimeError("interrupted after reservation")

    monkeypatch.setattr(resources, "reserve", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    assert resources.audit()["reserved_bytes"] == MAX_BYTES
    assert resources.audit()["retained_bytes"] == before["retained_bytes"]
    with resources._connect() as db:
        assert not any(
            KIND in row[0] for row in db.execute("SELECT descriptor FROM publications")
        )


def test_changed_caller_after_settlement_cannot_publish(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy.feature_resume_anchor import (
        KIND,
        publish_feature_anchor,
    )

    settle = resources.settle
    before = resources.audit()

    def changed(token):
        settle(token)
        qualified["sha256"] = "0" * 64

    monkeypatch.setattr(resources, "settle", changed)
    with pytest.raises(ValueError, match="context"):
        publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    after = resources.audit()
    assert after["retained_bytes"] > before["retained_bytes"]
    assert after["reserved_bytes"] == 0
    with resources._connect() as db:
        assert not any(
            KIND in row[0] for row in db.execute("SELECT descriptor FROM publications")
        )


@pytest.mark.parametrize(
    "fault,match",
    [
        ("schema", "schema"),
        ("rows", "metadata"),
        ("descriptor_size", "descriptor size"),
        ("json", "descriptor"),
        ("canonical", "canonical"),
        ("day", "day mismatch"),
        ("binding", "context"),
    ],
)
def test_malformed_anchor_artifacts_fail_closed(
    resources, qualified, days, fault, match
):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts, _encode
    from arblab.hyperliquid_copy.feature_resume_anchor import (
        FeatureAnchor,
        KIND,
        MAX_BYTES,
        SCHEMA,
        publish_feature_anchor,
    )

    valid = publish_feature_anchor(resources, qualified, days[1:], ["BTC"], SEMANTICS)
    inputs = valid.inputs
    inputs["window_sha256"] = "0" * 64  # A separate malformed fixture publication.
    schema = SCHEMA
    rows = [dict(day=d.day.isoformat(), inputs=_encode(d.inputs)) for d in days[1:]]
    if fault == "schema":
        schema, rows = (
            pa.schema([pa.field("wrong", pa.string())]),
            [dict(wrong="schema")],
        )
    elif fault == "rows":
        rows = [rows[0]] * 734
    elif fault == "descriptor_size":
        rows[0]["inputs"] = b" " * (64 * 1024 + 1)
    elif fault == "json":
        rows[0]["inputs"] = b"{"
    elif fault == "canonical":
        rows[0]["inputs"] = json.dumps(days[1].inputs, indent=2).encode()
    elif fault == "day":
        rows[0]["day"] = DAY.isoformat()
    relative = f"artifacts/{uuid.uuid4().hex}.parquet"
    token = resources.reserve(relative, MAX_BYTES, "payload")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), resources.root / relative)
    resources.settle(token)
    PublishedArtifacts(resources).publish(KIND, inputs, [token])
    before = resources.audit()
    with pytest.raises(ValueError, match=match):
        FeatureAnchor(resources, qualified, inputs)
    assert resources.audit() == before


def test_expired_lease_and_stale_anchor_engine_are_rejected(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_resume_anchor as module

    anchor = module.publish_feature_anchor(
        resources, qualified, days, ["BTC"], SEMANTICS
    )
    engine = module._engine
    monkeypatch.setattr(module, "_engine", lambda: "0" * 64)
    with pytest.raises(ValueError, match="engine"):
        module.FeatureAnchor(resources, qualified, anchor.inputs)
    with pytest.raises(ValueError, match="context"):
        anchor.verify()
    monkeypatch.setattr(module, "_engine", engine)
    resources.lease.__exit__(None)
    with pytest.raises(ValueError, match="lease"):
        anchor.verify()


@pytest.mark.parametrize("target", ["pin", "coins", "days"])
def test_caller_mutation_during_write_cannot_publish(
    resources, qualified, days, monkeypatch, target
):
    from arblab.hyperliquid_copy import feature_resume_anchor as module

    selected, coins = list(days), ["BTC"]
    original = module._write

    def changed(*args):
        result = original(*args)
        if target == "pin":
            qualified["sha256"] = "0" * 64
        elif target == "coins":
            coins.append("ETH")
        else:
            selected.pop()
        return result

    monkeypatch.setattr(module, "_write", changed)
    with pytest.raises(ValueError, match="context"):
        module.publish_feature_anchor(resources, qualified, selected, coins, SEMANTICS)
    with resources._connect() as db:
        assert not any(
            module.KIND in r[0]
            for r in db.execute("SELECT descriptor FROM publications")
        )


def test_source_change_during_final_engine_check_is_rejected(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_resume_anchor as module

    original, write = module._engine, module._write
    written = False

    def completed(*args):
        nonlocal written
        result = write(*args)
        written = True
        return result

    def changed():
        result = original()
        if written:
            with Path(qualified["path"]).open("ab") as handle:
                handle.write(b"changed")
        return result

    monkeypatch.setattr(module, "_write", completed)
    monkeypatch.setattr(module, "_engine", changed)
    with pytest.raises(ValueError):
        module.publish_feature_anchor(resources, qualified, days, ["BTC"], SEMANTICS)
    with resources._connect() as db:
        assert not any(
            module.KIND in r[0]
            for r in db.execute("SELECT descriptor FROM publications")
        )


def test_anchor_writer_batches_are_bounded_and_charge_actual_bytes(resources, days):
    from types import SimpleNamespace
    from arblab.hyperliquid_copy import feature_resume_anchor as module

    # Exercise the writer's batching independently of chain qualification.
    token = module._write(resources, SimpleNamespace(days=[days[0]] * 17))
    with resources._connect() as db:
        relative, size = db.execute(
            "SELECT path,bytes FROM allocations WHERE token=?", (token,)
        ).fetchone()
    with pq.ParquetFile(resources.root / relative) as reader:
        assert reader.metadata.num_rows == 17
        assert [
            reader.metadata.row_group(i).num_rows
            for i in range(reader.metadata.num_row_groups)
        ] == [8, 8, 1]
        assert reader.schema_arrow == module.SCHEMA
    assert size == (resources.root / relative).stat().st_size < module.MAX_BYTES
    assert resources.audit()["reserved_bytes"] == 0


def test_output_byte_cap_retains_partial_obligation(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_resume_anchor as module

    monkeypatch.setattr(module, "MAX_BYTES", 10)
    with pytest.raises(ValueError, match="byte limit"):
        module.publish_feature_anchor(resources, qualified, days, ["BTC"], SEMANTICS)
    assert resources.audit()["reserved_bytes"] == 10
    with resources._connect() as db:
        assert not any(
            module.KIND in r[0]
            for r in db.execute("SELECT descriptor FROM publications")
        )


def test_insufficient_shared_budget_prevents_writer_open(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy import feature_resume_anchor as module

    remaining = resources.limit - resources.audit()["total_bytes"]
    resources.reserve(
        "staging/" + "a" * 32, remaining - module.MAX_BYTES + 1, "payload"
    )
    before = resources.audit()

    def forbidden(*args, **kwargs):
        pytest.fail("writer opened before budget approval")

    monkeypatch.setattr(module.pq, "ParquetWriter", forbidden)
    with pytest.raises(ValueError, match="budget"):
        module.publish_feature_anchor(resources, qualified, days, ["BTC"], SEMANTICS)
    assert resources.audit() == before


def test_old_lease_feature_objects_cannot_be_adopted(resources, qualified, days):
    from arblab.hyperliquid_copy.derived_cache_lease import CacheLease
    from arblab.hyperliquid_copy.derived_cache_resources import CacheResources
    from arblab.hyperliquid_copy.feature_resume_anchor import publish_feature_anchor

    before = resources.audit()
    resources.lease.__exit__(None)
    with CacheLease(resources.root) as lease:
        current = CacheResources(lease, resources.identity)
        with pytest.raises(ValueError, match="lease"):
            publish_feature_anchor(current, qualified, days, ["BTC"], SEMANTICS)
        assert current.audit() == before


def test_compressed_oversized_descriptor_rejected_before_decoding(
    resources, qualified, days, monkeypatch
):
    from arblab.hyperliquid_copy.derived_publication import PublishedArtifacts
    from arblab.hyperliquid_copy import feature_resume_anchor as module

    anchor = module.publish_feature_anchor(
        resources, qualified, days, ["BTC"], SEMANTICS
    )
    inputs = anchor.inputs
    inputs["window_sha256"] = "0" * 64
    relative = f"artifacts/{uuid.uuid4().hex}.parquet"
    token = resources.reserve(relative, module.MAX_BYTES, "payload")
    table = pa.Table.from_pylist(
        [dict(day=DAY.isoformat(), inputs=b"x" * (2 * 1024**2))], schema=module.SCHEMA
    )
    pq.write_table(table, resources.root / relative, compression="zstd")
    resources.settle(token)
    PublishedArtifacts(resources).publish(module.KIND, inputs, [token])
    metadata = pq.read_metadata(resources.root / relative)
    assert (resources.root / relative).stat().st_size < 1024**2
    assert metadata.row_group(0).total_byte_size > 1024**2

    def forbidden(*args, **kwargs):
        pytest.fail("oversized row group reached decompression")

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", forbidden)
    with pytest.raises(ValueError, match="row group"):
        module.FeatureAnchor(resources, qualified, inputs)


@pytest.mark.parametrize("operation", ["verify", "publish"])
def test_lease_expiring_during_final_engine_hash_is_rejected(
    resources, qualified, days, monkeypatch, operation
):
    from arblab.hyperliquid_copy import feature_resume_anchor as module

    anchor = module.publish_feature_anchor(
        resources, qualified, days, ["BTC"], SEMANTICS
    )
    engine, constructor = module._engine, module.FeatureAnchor.__init__
    armed = operation == "verify"

    def completed(self, *args):
        nonlocal armed
        constructor(self, *args)
        armed = True

    def expired():
        value = engine()
        if armed:
            resources.lease.__exit__(None)
        return value

    monkeypatch.setattr(module.FeatureAnchor, "__init__", completed)
    monkeypatch.setattr(module, "_engine", expired)
    with pytest.raises(ValueError, match="lease"):
        if operation == "verify":
            anchor.verify()
        else:
            module.publish_feature_anchor(
                resources, qualified, days, ["BTC"], SEMANTICS
            )
