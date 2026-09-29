"""Limited saved-ledger checks; never promote research qualification."""

from .repository import number


def metric(value, unit="ratio", reason="Not provided or undefined"):
    value = number(value)
    return dict(value=value, unit=unit, reason=reason if value is None else None)


def ledger_sum(rows, field, sign=1):
    if rows is None or any(number(r.get(field)) is None for r in rows):
        return None
    return sign * sum(float(r[field]) for r in rows)


def accounting(rows, summary, fills, funding):
    initial = number(rows[0].get("equity")) if rows else None
    final = number(rows[-1].get("equity")) if rows else None
    fees, funding_cost = ledger_sum(fills, "fee"), ledger_sum(funding, "cash_delta", -1)
    output = {"fees_usd": metric(fees, "usd"), "funding_usd": metric(funding_cost, "usd")}
    checks = []

    def check(name, left, right):
        delta = left - right if left is not None and right is not None else None
        checks.append(dict(name=name, difference_usd=delta,
                           passed=abs(delta) <= .01 if delta is not None else None,
                           reason="Missing or invalid accounting evidence" if delta is None else None))

    components = [number(summary.get(k)) for k in ("final_collateral", "final_unrealized_pnl")]
    check("Final equity = stored collateral + unrealized PnL", final,
          sum(components) if all(v is not None for v in components) else None)
    check("Final curve equity = stored final equity", final, number(summary.get("final_equity")))
    differences = []
    for row in rows:
        values = [number(row.get(k)) for k in ("equity", "cash", "unrealized_pnl")]
        if any(v is None for v in values):
            differences = []
            break
        differences.append(abs(values[0] - values[1] - values[2]))
    check("Every equity row = cash + unrealized PnL", max(differences) if differences else None, 0)
    for title, observed, key in (("Fill fees = summary fees", fees, "fee_drag"),
                                 ("Funding ledger = summary funding cost", funding_cost, "funding_drag")):
        drag = number(summary.get(key))
        check(title, observed, initial * drag if initial is not None and drag is not None else None)
    for key in ("final_equity", "final_collateral", "final_unrealized_pnl", "residual_gross_exposure",
                "residual_net_exposure", "max_gross_leverage", "turnover", "liquidation_count",
                "stale_requests", "unexecuted_requests", "superseded_requests", "below_threshold_requests",
                "max_mark_age_seconds"):
        unit = "usd" if key.startswith(("final_", "residual_")) else "count" if key.endswith(("count", "requests", "seconds")) else "ratio"
        output[key] = metric(summary.get(key), unit)
    output["executed_fills"] = metric(len(fills) if fills is not None else None, "count")
    return output, checks


def worst_context(weeks, fills, funding, equity_rows):
    output = []
    for week in sorted((w for w in weeks if not w["partial"]), key=lambda w: w["return_value"])[:5]:
        # Events exactly at the prior equity endpoint are already in that mark.
        trades = [r for r in fills if week["start"] < r["time"] <= week["end"]] if fills is not None else None
        charges = [r for r in funding if week["start"] < r["time"] <= week["end"]] if funding is not None else None
        marks = [r for r in equity_rows if week["start"] < r["time"] <= week["end"]]
        valid = bool(marks) and all(number(r.get(k)) is not None for r in marks for k in ("gross_exposure", "net_exposure"))
        output.append(dict(start=week["start"], end=week["end"], return_value=week["return_value"],
                           fills=len(trades) if trades is not None else None,
                           fees_usd=ledger_sum(trades, "fee"), funding_usd=ledger_sum(charges, "cash_delta", -1),
                           max_gross_usd=max(r["gross_exposure"] for r in marks) if valid else None,
                           mean_net_usd=sum(r["net_exposure"] for r in marks) / len(marks) if valid else None,
                           exposure_reason=None if valid else "Missing or invalid full-resolution exposure",
                           assets=sorted({str(r["coin"]) for r in (trades or []) if r.get("coin")})))
    return output
