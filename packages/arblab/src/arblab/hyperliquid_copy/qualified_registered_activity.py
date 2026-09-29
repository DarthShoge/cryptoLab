"""Own one existing cache lease for a registered qualified reader's lifetime."""

from pathlib import Path

from .derived_cache_lease import CacheLease
from .derived_cache_resources import CacheResources
from .derived_cache_policy import open_expanded_cache
from .derived_cache_policy_64 import open_expanded_64_cache
from .derived_publication import _encode
from .download import file_hash
from .qualified_registration import (
    cache_reference,
    verify_qualified_registration,
    POLICY,
)
from .qualified_scheduled_activity import QualifiedScheduledActivity
from .qualified_source_session import _identity


def _engine():
    root = Path(__file__).parent
    return tuple(
        file_hash(root / name)
        for name in (
            "qualified_registered_activity.py",
            "qualified_registration.py",
            "feature_history_policy.py",
            "ranking_staging_policy.py",
            "derived_cache_policy.py",
            "derived_cache_expansion.py",
            "derived_cache_policy_64.py",
            "derived_cache_expansion_64.py",
        )
    )


class _OwnedActivity:
    def __init__(self, manifest, activity, lease, binding, frozen, engine):
        self._manifest, self._activity, self._lease = manifest, activity, lease
        self._binding, self._frozen, self._engine = binding, frozen, engine
        self.closed = False

    def __getattr__(self, name):
        if name not in {
            "prepare",
            "observed",
            "positions",
            "position",
            "volume",
            "hourly_exposure",
            "rank",
        }:
            raise AttributeError(name)
        return getattr(self._activity, name)

    def close(self):
        if self.closed:
            return
        if self._activity._busy:
            raise ValueError("Cannot close busy qualified registered activity")
        try:
            self._activity._verify()
            self._activity.close()
            _verify_disk_context(self._manifest, self._frozen)
            if (
                verify_qualified_registration(self._manifest) != self._binding
                or _context(self._manifest) != self._frozen
                or _engine() != self._engine
            ):
                raise ValueError("Qualified registered activity context changed")
            reference = self._binding[1]
            if "expansion_receipt" in reference:
                _open_resources(self._lease, reference)
            _verify_disk_context(self._manifest, self._frozen)
            if _context(self._manifest) != self._frozen or _engine() != self._engine:
                raise ValueError("Qualified registered activity context changed")
        finally:
            try:
                self._activity.close()
            finally:
                self.closed = True
                self._lease.__exit__(None, None, None)

    def __enter__(self):
        if self.closed:
            raise ValueError("Qualified registered activity is closed")
        return self

    def __exit__(self, *_):
        self.close()


def _context(manifest):
    return _encode(dict(directory=str(manifest.directory), metadata=manifest.metadata))


def _verify_disk_context(manifest, frozen):
    # Local import avoids the dataset loader's module-initialization cycle.
    # Reparse bounded metadata, without another full canonical-corpus hash scan.
    from .proxy_dataset import ProxyDatasetManifest

    current = ProxyDatasetManifest(manifest.directory)
    if _context(current) != frozen:
        raise ValueError("Qualified registered manifest changed on disk")


def load_qualified_activity(manifest, config):
    """Borrow the specified existing envelope; never create/recover another cache."""
    frozen, engine = _context(manifest), _engine()
    reference = cache_reference(manifest.metadata.get("derived_cache"))
    lease = CacheLease(reference["path"])
    lease.__enter__()
    activity = None
    try:
        _verify_disk_context(manifest, frozen)
        resources = _open_resources(lease, reference)
        pin, checked_reference = verify_qualified_registration(manifest)
        if checked_reference != reference:
            raise ValueError("Registered cache reference changed during load")
        manifest_path = manifest.directory / "manifest.json"
        manifest_identity = _identity(manifest_path)

        def validate_registration():
            if _engine() != engine:
                raise ValueError("Qualified registration engine changed during query")
            _verify_disk_context(manifest, frozen)
            if (
                _context(manifest) != frozen
                or _identity(manifest_path) != manifest_identity
            ):
                raise ValueError("Qualified registration context changed during query")

        activity = QualifiedScheduledActivity(
            resources,
            pin,
            config,
            coverage_start=manifest.metadata["coverage_start"],
            coverage_end=manifest.metadata["coverage_end"],
            semantics=manifest.metadata["fee_semantics"],
            history_policy=POLICY,
            feature_history_policy=manifest.metadata.get("feature_history_policy"),
            ranking_staging_policy=manifest.metadata.get("ranking_staging_policy"),
            execution_policy_name=manifest.metadata.get("execution_policy"),
            validation=validate_registration,
        )
        if _context(manifest) != frozen or _engine() != engine:
            raise ValueError("Qualified registration context changed during load")
        _verify_disk_context(manifest, frozen)
        return _OwnedActivity(
            manifest, activity, lease, (pin, checked_reference), frozen, engine
        )
    except BaseException:
        try:
            if activity is not None:
                activity.close()
        finally:
            lease.__exit__(None, None, None)
        raise


def _open_resources(lease, reference):
    if "expansion_64gib_receipt" in reference:
        return open_expanded_64_cache(
            lease,
            reference["identity"],
            reference["expansion_receipt"],
            reference["expansion_64gib_receipt"],
        )
    if "expansion_receipt" in reference:
        return open_expanded_cache(
            lease, reference["identity"], reference["expansion_receipt"]
        )
    return CacheResources(lease, reference["identity"])
