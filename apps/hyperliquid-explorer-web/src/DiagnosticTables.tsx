import type { components } from "./api.generated";
import { formatValue, label, utc } from "./format";

export type DiagnosticData = components["schemas"]["Diagnostics"];
export type DiagnosticMetric = components["schemas"]["DiagnosticMetric"];

const names: Record<string, string> = {
  count: "Complete weeks", paired_count: "Paired BTC weeks",
  annualized_weekly_volatility: "Annualized volatility (weekly returns)",
  mean_ci_low: "Mean weekly return · 95% interval lower",
  mean_ci_high: "Mean weekly return · 95% interval upper",
  excess_ci_low: "Mean weekly excess · 95% interval lower",
  excess_ci_high: "Mean weekly excess · 95% interval upper",
  excess_total_return_pp: "Total-return outperformance (percentage points, not alpha)",
  max_mark_age_seconds: "Maximum observed mark age (seconds)",
  liquidation_count: "Recorded simulated liquidations (not a realism guarantee)",
};
export function MetricValue({ metric }: { metric?: DiagnosticMetric }) {
  return <><strong>{formatValue(metric?.value, metric?.unit ?? "ratio")}</strong>{metric?.reason && <small>{metric.reason}</small>}</>;
}
export function MetricTable({ values, benchmark }: { values: Record<string, DiagnosticMetric>; benchmark?: Record<string, DiagnosticMetric> }) {
  return <div className="table-scroll"><table><thead><tr><th>Metric</th><th>Strategy</th>{benchmark && <th>BTC perpetual control</th>}</tr></thead><tbody>
    {Object.entries(values).map(([key, value]) => <tr key={key}><th scope="row">{names[key] ?? label(key)}</th><td><MetricValue metric={value} /></td>{benchmark && <td><MetricValue metric={benchmark[key]} /></td>}</tr>)}
  </tbody></table></div>;
}

export function DiagnosticAccounting({ data }: { data: DiagnosticData }) {
  return <section className="panel"><h2>Accounting & execution</h2>
    <p className="muted">Local arithmetic checks only; original reconciliation status is unchanged. Open positions are not force-closed. Fees and funding are already included in equity. Negative funding cost is a credit.</p>
    <MetricTable values={data.accounting} />
    <h3>Internal consistency checks · $0.01 tolerance</h3>
    <div className="table-scroll"><table><thead><tr><th>Check</th><th>Result</th><th>Difference</th></tr></thead><tbody>{data.checks.map(c => <tr key={c.name}><th scope="row">{c.name}</th><td>{c.passed == null ? "Unavailable" : c.passed ? "Pass" : "MISMATCH"}{c.reason && <small>{c.reason}</small>}</td><td>{formatValue(c.difference_usd, "usd")}</td></tr>)}</tbody></table></div>
    <h3>Ending positions · saved native or proxy units</h3>
    <p className="muted">For hourly proxy execution, quantities and entries are in proxy units—not necessarily native Hyperliquid contract units. Independent per-asset PnL attribution is unavailable here.</p>
    <pre>{JSON.stringify(data.positions, null, 2)}</pre>
    <h3>Worst complete weeks · execution context</h3>
    <p className="muted">Dated events provide context, not proof of what caused losses. Interval excludes the starting equity mark and includes the ending mark. Maximum gross and mean signed net exposure use every validated source observation in that interval, in USD.</p>
    {data.worst_weeks.length ? <div className="table-scroll"><table><thead><tr><th>Week start (UTC)</th><th>Return</th><th>Fills</th><th>Fees</th><th>Funding cost</th><th>Max gross USD</th><th>Mean net USD</th><th>Traded assets</th></tr></thead><tbody>{data.worst_weeks.map(w => <tr key={w.start}><td>{utc(w.start)}</td><td>{formatValue(w.return_value, "percent")}</td><td>{formatValue(w.fills, "count")}</td><td>{formatValue(w.fees_usd, "usd")}</td><td>{formatValue(w.funding_usd, "usd")}</td><td>{formatValue(w.max_gross_usd, "usd")}{w.exposure_reason && <small>{w.exposure_reason}</small>}</td><td>{formatValue(w.mean_net_usd, "usd")}</td><td>{w.assets?.join(", ") || "No trades / unavailable"}</td></tr>)}</tbody></table></div> : <p>No complete weeks available.</p>}
  </section>;
}

export function DiagnosticCohorts({ data, onTraders }: { data: DiagnosticData; onTraders: (date: string, instrument: string) => void }) {
  return <section className="panel"><h2>Historical cohort composition</h2>
    <p className="notice">Coverage can start on different dates for each asset. Missing dates are not zero-member cohorts. Inspect a date to see selected traders, exclusions and wallet history.</p>
    <p className="muted">Saved turnover = (entries + exits) / (previous + current members). Unknown prior cohort: N/A. Known empty → nonempty: 100%; both known empty: 0%.</p>
    {data.cohort_reason && <p>{data.cohort_reason}</p>}
    <div className="table-scroll" style={{ maxHeight: 440 }}><table><thead><tr><th>Decision (UTC)</th><th>Asset</th><th>Candidates</th><th>Eligible</th><th>Selected</th><th>Turnover</th><th>Drilldown</th></tr></thead><tbody>{data.cohorts.map((c, i) => <tr key={i}><td>{utc(c.decision_time)}</td><td>{c.coin ?? "Pooled"}</td><td>{formatValue(c.candidate_count, "count")}</td><td>{formatValue(c.eligible_count, "count")}</td><td>{formatValue(c.selected_count, "count")}</td><td>{formatValue(c.membership_turnover, "percent")}</td><td><button aria-label={`Inspect cohort ${c.coin ?? "pooled"} ${c.decision_time}`} onClick={() => onTraders(c.decision_time.slice(0, 10), c.coin ?? "pooled")}>Inspect cohort</button></td></tr>)}</tbody></table></div>
  </section>;
}
