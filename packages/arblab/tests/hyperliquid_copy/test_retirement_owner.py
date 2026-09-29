"""Authenticated operation ownership; all disposal is confined to fixtures."""

import pytest

from .test_cache_retirement_inventory import cache
from .test_cache_retirement import prepare, rewrite_intent


def owned(resources, owner="a" * 64):
    from arblab.hyperliquid_copy.cache_retirement import prepare_retirement

    return prepare_retirement(
        resources,
        "candidate_metrics",
        {"source": "fixture"},
        reason="Fixture rolling owner",
        owner=owner,
    )


def test_owner_is_bound_in_receipt_and_journal_and_survives_finish(cache):
    from arblab.hyperliquid_copy.cache_retirement import (
        begin_retirement,
        finish_retirement,
    )
    from arblab.hyperliquid_copy.cache_retirement_journal import load_journal

    resources, target, other = cache
    inputs = owned(resources)
    body, _ = load_journal(resources, inputs)
    assert inputs["owner"] == body["owner"] == "a" * 64
    begin_retirement(resources, inputs)
    result = finish_retirement(resources, inputs)
    assert result["target"] == target.key
    assert load_journal(resources, inputs)[0]["owner"] == "a" * 64


@pytest.mark.parametrize("owner", [True, {}, [], "", "a" * 63, "A" * 64, 1])
def test_invalid_owner_rejects_before_any_allocation(cache, owner):
    resources, _, _ = cache
    before = resources.audit()
    with pytest.raises(ValueError, match="owner"):
        owned(resources, owner)
    assert resources.audit() == before


def test_unowned_receipt_keeps_existing_shape(cache):
    from arblab.hyperliquid_copy.cache_retirement_journal import load_journal

    resources, _, _ = cache
    inputs = prepare(resources)
    body, _ = load_journal(resources, inputs)
    assert "owner" not in inputs and "owner" not in body


@pytest.mark.parametrize("mutation", ["change", "remove"])
def test_rehashed_journal_cannot_change_or_remove_owner(cache, mutation):
    from arblab.hyperliquid_copy.cache_retirement import begin_retirement

    resources, _, _ = cache
    inputs = owned(resources)

    def change(body):
        if mutation == "change":
            body["owner"] = "b" * 64
        else:
            del body["owner"]

    forged = rewrite_intent(resources, inputs, change)
    before = resources.audit()
    with pytest.raises(ValueError, match="owner|structure"):
        begin_retirement(resources, forged)
    assert resources.audit() == before
