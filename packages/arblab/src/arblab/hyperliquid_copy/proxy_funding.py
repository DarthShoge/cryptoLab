"""Historical funding with actual settlement time and separate coverage buckets."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .contracts import finite, symbol, utc


@dataclass(frozen=True)
class FundingEvent:
    instrument_id: str
    time: datetime
    hour: datetime
    rate: float
    premium: float


@dataclass(frozen=True)
class FundingHistory:
    events: tuple[FundingEvent, ...]
    missing_hours: tuple[datetime, ...]


def funding_history(instrument_id, start, end, fetch_page):
    symbol(instrument_id)
    start, end = utc(start), utc(end)
    if not 0 < (end - start).total_seconds() <= 93 * 86400:
        raise ValueError("Funding range must be positive and at most 93 days")
    if any(t.minute or t.second or t.microsecond for t in (start, end)):
        raise ValueError("Funding coverage requires aligned UTC hours")
    begin, finish = int(start.timestamp() * 1000), int(end.timestamp() * 1000)
    cursor = begin
    events, hours = {}, {}
    for _ in range(100):
        rows = fetch_page(
            dict(
                type="fundingHistory",
                coin=instrument_id,
                startTime=cursor,
                endTime=finish - 1,
            )
        )
        if not isinstance(rows, list) or len(rows) > 5000:
            raise ValueError("Invalid funding response")
        if not rows:
            break
        latest = cursor - 1
        for row in rows:
            timestamp = row["time"]
            if type(timestamp) is not int or not begin <= timestamp < finish:
                raise ValueError("Invalid funding settlement timestamp")
            if row["coin"] != instrument_id:
                raise ValueError("Unexpected funding instrument")
            at = datetime.fromtimestamp(timestamp / 1000, timezone.utc)
            hour = at.replace(minute=0, second=0, microsecond=0)
            event = FundingEvent(
                instrument_id,
                at,
                hour,
                finite(row["fundingRate"]),
                finite(row["premium"]),
            )
            if timestamp in events and events[timestamp] != event:
                raise ValueError("conflicting funding settlement")
            if hour in hours and hours[hour] != timestamp:
                raise ValueError("Multiple funding settlements in one hour")
            events[timestamp], hours[hour] = event, timestamp
            latest = max(latest, timestamp)
        if latest < cursor:
            raise ValueError("Funding pagination did not advance")
        cursor = latest + 1
        if cursor >= finish:
            break
    else:
        raise ValueError("Funding pagination exceeds request ceiling")
    expected = {
        start + timedelta(hours=i)
        for i in range(int((end - start).total_seconds() / 3600))
    }
    return FundingHistory(
        tuple(events[t] for t in sorted(events)), tuple(sorted(expected - hours.keys()))
    )
