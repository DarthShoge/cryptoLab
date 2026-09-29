"""Stable Arrow schemas, including genuinely empty selection histories."""

import pyarrow as pa

TIME = pa.timestamp("us", tz="UTC")
TEXT = pa.string()
NUMBER = pa.float64()
INTEGER = pa.int64()
STRINGS = pa.list_(TEXT)
MARKET_RANKINGS = pa.schema(
    {
        "decision_time": TIME,
        "effective_at": TIME,
        "instrument_id": TEXT,
        "display_name": TEXT,
        "venue": TEXT,
        "asset_class": TEXT,
        "availability_basis": TEXT,
        "proxy_ticker": TEXT,
        "volume_usd": NUMBER,
        "rank": INTEGER,
        "eligible": pa.bool_(),
        "selected": pa.bool_(),
        "reasons": STRINGS,
        "budget": NUMBER,
        "window_start": TIME,
        "window_end": TIME,
    }
)
COHORT_FIELDS = {
    "decision_time": TIME,
    "members": STRINGS,
    "entries": STRINGS,
    "exits": STRINGS,
    "candidate_count": INTEGER,
    "eligible_count": INTEGER,
    "selected_count": INTEGER,
    "requested_count": INTEGER,
    "retention": NUMBER,
    "membership_turnover": NUMBER,
}
MARKET_COHORTS = pa.schema(COHORT_FIELDS | {"effective_at": TIME})
TRADER_COHORTS = pa.schema(
    COHORT_FIELDS
    | {
        "coin": TEXT,
        "cutoff_address": TEXT,
        "market_decision_time": TIME,
        "decision_trigger": TEXT,
    }
)
EMPTY_RANKINGS = pa.schema(
    {
        "decision_time": TIME,
        "coin": TEXT,
        "user": TEXT,
        "rank": INTEGER,
        "score": NUMBER,
        "selected": pa.bool_(),
        "eligible": pa.bool_(),
        "reasons": STRINGS,
        "market_decision_time": TIME,
        "decision_trigger": TEXT,
    }
)
EMPTY_CONTRIBUTIONS = pa.schema(
    {
        "time": TIME,
        "decision_time": TIME,
        "market_decision_time": TIME,
        "coin": TEXT,
        "user": TEXT,
        "target_contribution": NUMBER,
    }
)
