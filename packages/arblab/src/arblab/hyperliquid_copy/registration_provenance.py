"""Verify the immutable offline provenance copies of annual registrations."""

import re

from .download import file_hash

NAMES = frozenset(
    (
        "price_source.json",
        "funding_source.json",
        "activity_qualification.json",
        "activity_source.json",
    )
)
MAX_BYTES = 16 * 1024**2


def verify_registration_provenance(directory, metadata):
    if (
        "registration_engine" not in metadata
        and "registration_provenance" not in metadata
    ):
        return  # Legacy datasets do not claim this publication contract.
    inventory = metadata.get("registration_provenance")
    if not isinstance(inventory, dict) or set(inventory) != NAMES:
        raise ValueError("Invalid registration provenance inventory")
    expected = {
        "price_source.json": metadata["price_source_manifest_hash"],
        "funding_source.json": metadata["funding_source_manifest_hash"],
        "activity_qualification.json": metadata["activity_provenance"][
            "source_manifest_hash"
        ],
    }
    for name, digest in inventory.items():
        if (
            not isinstance(digest, str)
            or not re.fullmatch("[a-f0-9]{64}", digest)
            or name in expected
            and digest != expected[name]
        ):
            raise ValueError("Invalid registration provenance identity binding")
        path = directory / name
        if (
            path.is_symlink()
            or not path.is_file()
            or not 0 < path.stat().st_size <= MAX_BYTES
        ):
            raise ValueError("Missing/unsafe registration provenance copy")
        if file_hash(path) != digest:
            raise ValueError("Registration provenance copy changed")
