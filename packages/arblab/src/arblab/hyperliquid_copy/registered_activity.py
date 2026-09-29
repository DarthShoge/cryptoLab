"""Bridge immutable registered datasets to disposable rolling query caches."""

from datetime import timedelta

from .activity_checkpoints import ActivityCheckpoints, _engine, _verify
from .compact_catalog import event_bounds
from .contracts import semantic_hash
from .lab_config import day
from .proxy_activity import ProxyActivity
from .proxy_availability import validate_native_fills
from .scheduled_activity import ScheduledActivity, history_days
from .sharded_validation import qualified_seed


def build_initial(store, manifest, cutoff, key, *, temp_root):
    # Own the full reader here: a caller cannot present a partially filtered or
    # differently sourced reader as proof that the registered corpus is valid.
    manifest.verify()
    entries = []
    for entry in manifest.metadata["files"]:
        if not entry["name"].startswith("fills-"):
            continue
        path = manifest.paths[entry["name"]]
        low, high = event_bounds(path)
        entries.append(
            dict(
                entry,
                path=str(path),
                min_time=low,
                max_time=high,
                manifest_id=manifest.identity,
            )
        )
    replay_entries = [entry for entry in entries if entry["min_time"] is not None]
    if manifest.validation_mode == "sharded_v1":
        with qualified_seed(
            entries,
            set(manifest.metadata["coins"]),
            manifest.coverage_start,
            manifest.coverage_end,
            cutoff,
            temp_root=temp_root,
            native_starts=manifest.native_starts,
        ) as seed:
            manifest.verify()
            with ProxyActivity([seed], temp_root=temp_root) as activity:
                return store._publish(
                    activity, cutoff, replay_entries, [manifest.identity], cache_key=key
                )
    with ProxyActivity(manifest.fill_paths, temp_root=temp_root) as activity:
        activity.validate_registered_scope(
            set(manifest.metadata["coins"]),
            manifest.coverage_start,
            manifest.coverage_end,
        )
        validate_native_fills(activity, manifest.native_starts)
        manifest.verify()
        return store._publish(
            activity, cutoff, replay_entries, [manifest.identity], cache_key=key
        )


def load_registered_activity(manifest, config, *, temp_root):
    cutoff = day(config.start) - timedelta(days=history_days(config))
    if cutoff < manifest.coverage_start or day(config.end) > manifest.coverage_end:
        raise ValueError("Missing required proxy trader/market warmup")
    store = ActivityCheckpoints(manifest.directory / ".activity_checkpoints")
    key = semantic_hash(
        dict(dataset=manifest.identity, cutoff=cutoff, engine=_engine())
    )
    identity = store.lookup(key)
    if identity is None:
        identity = build_initial(store, manifest, cutoff, key, temp_root=temp_root)
    meta = store.metadata(identity)
    directory = store.root / identity
    if directory.is_symlink():
        raise ValueError("Checkpoint directory identity changed")
    _verify([dict(meta["seed"], path=str(directory / "seeds.parquet"))])
    return ScheduledActivity(store, identity, config, temp_root=temp_root)
