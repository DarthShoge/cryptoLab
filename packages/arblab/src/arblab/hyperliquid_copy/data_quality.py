"""Fail-closed validation and location-independent dataset identities."""
from collections import Counter
from dataclasses import dataclass
from datetime import timedelta
from math import isclose

from .contracts import address, finite, semantic_hash, symbol, utc


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    count: int
    detail: str


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    issues: tuple
    dataset_sha256: str = ""

    def assert_accepted(self):
        if not self.accepted:
            raise ValueError("data quality: " + "; ".join(f"{i.code}: {i.detail}" for i in self.issues if i.severity == "error"))


def dataset_hash(manifests):
    rows = [{k:v for k,v in row.items() if k not in {"local_path", "root", "created_at", "downloaded_at"}}
            for row in manifests]
    return semantic_hash(sorted(rows, key=semantic_hash))


def validate_fills(fills, start, end):
    utc(start), utc(end)
    counts, seen, positions = Counter(), set(), {}
    valid = []
    for f in fills:
        try:
            address(f.user), symbol(f.coin), utc(f.exchange_time), utc(f.ingested_at)
            for value in (f.px, f.sz, f.start_position, f.post_position, f.closed_pnl, f.fee):
                finite(value)
            if f.side not in ("A", "B") or f.px <= 0 or f.sz <= 0:
                raise ValueError("invalid side/price/size")
        except (ValueError, TypeError):
            counts["invalid_data"] += 1
            continue
        valid.append(f)
    for f in sorted(valid, key=lambda f:f.order_key):
        if f.event_id in seen:
            counts["duplicate_event_id"] += 1
        seen.add(f.event_id)
        if not start <= f.exchange_time < end:
            counts["outside_window"] += 1
        post = f.start_position + (f.sz if f.side == "B" else -f.sz)
        if not isclose(post, f.post_position, abs_tol=1e-8):
            counts["invalid_position_delta"] += 1
        key = f.user, f.coin
        if key in positions and not isclose(positions[key], f.start_position, abs_tol=1e-8):
            counts["position_discontinuity"] += 1
        positions[key] = f.post_position
        expected = ("Long > Short" if f.start_position > 0 and post < 0 else
                    "Short > Long" if f.start_position < 0 and post > 0 else
                    "Close Long" if f.start_position > 0 and f.side == "A" else
                    "Close Short" if f.start_position < 0 and f.side == "B" else
                    "Open Long" if f.side == "B" else "Open Short")
        if f.direction != expected:
            counts["invalid_direction"] += 1
    issues = tuple(QualityIssue(code, "error", n, code) for code,n in sorted(counts.items()))
    return QualityReport(not issues, issues)


def validate_market(market, coins, start, end, *, research, warmup_start=None, l2_threshold=.95, funding_threshold=.95):
    start, end = utc(start), utc(end)
    issues = []
    for coin in coins:
        for code, first, step, lookup in [
            ("mark_coverage", warmup_start or start, timedelta(minutes=1), lambda t:market.mark(coin,t)),
            ("execution_coverage", start, timedelta(seconds=1), lambda t:market.execution_book(coin,t)),
        ]:
            at, count, missing = first, 0, 0
            while at <= end:
                count += 1
                try:
                    missing += lookup(at) is None
                except ValueError:
                    missing += 1
                at += step
            if missing/count > 1-l2_threshold+1e-12:
                issues.append(QualityIssue(code, "error", missing, f"{coin}: {count-missing}/{count}"))
        at = start.replace(minute=0, second=0, microsecond=0)+timedelta(hours=1)
        count, missing = 0, 0
        while at <= end:
            count += 1
            missing += (coin, at) not in market.funding
            at += timedelta(hours=1)
        threshold = 1. if research else funding_threshold
        if missing:
            severity = "error" if missing/count > 1-threshold+1e-12 else "warning"
            issues.append(QualityIssue("funding_coverage", severity, missing, f"{coin}: {count-missing}/{count}; no imputation allowed"))
    return QualityReport(not any(i.severity == "error" for i in issues), tuple(issues))
