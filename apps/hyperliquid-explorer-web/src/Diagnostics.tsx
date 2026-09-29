import { useResource } from "./api";
import { Status } from "./Status";
import { formatValue, label } from "./format";
import { DiagnosticProfiles } from "./DiagnosticProfiles";
import { DiagnosticAccounting, DiagnosticCohorts, MetricTable, MetricValue, type DiagnosticData } from "./DiagnosticTables";

function object(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : {};
}
function display(value: unknown): string {
  if (value == null) return "Not provided";
  if (Array.isArray(value)) return value.map(display).join(", ");
  if (typeof value === "object") return Object.entries(value).map(([k, v]) => `${label(k)}: ${display(v)}`).join(" · ");
  return String(value);
}
function SystemCard({ data }: { data: DiagnosticData }) {
  const config = data.config, follower = object(config.follower ?? config), trader = object(config.trader ?? config), markets = object(config.market_universe), proxy = object(config.proxy);
  const fields: [string, unknown][] = [
    ["Saved hypothesis", data.name], ["Configured period (end exclusive)", `${config.start} → ${config.end}`],
    ["Starting capital", formatValue(Number(follower.initial_equity), "usd")],
    ["Requested markets", markets.instrument_ids ?? config.coins], ["Asset classes / mode", [markets.classes, markets.mode]],
    ["Market allocation", markets.allocation], ["Trader ranking scope", trader.scope],
    ["Trailing lookback (days)", trader.lookback_days], ["Selection", trader.selection === "fraction" ? `Top ${Number(trader.top_fraction) * 100}%` : `Top ${trader.top_n}`],
    ["Cohort bounds", `${trader.min_cohort}–${trader.max_cohort}`],
    ["Eligibility", Object.fromEntries(Object.entries(trader).filter(([k]) => k.startsWith("min_") && k !== "min_cohort"))],
    ["Ranking weights", trader.metric_weights], ["Ranking preference", trader.metric_directions],
    ["Trader selection / position schedule", [trader.reselection, config.rebalance ?? `${follower.update_minutes} minutes`]],
    ["Copying rule", follower.aggregation], ["Gross / per-asset caps", [follower.gross_cap, follower.asset_cap]],
    ["Minimum known traders / weight", [follower.min_known, follower.min_known_weight]],
    ["Fee / slippage (bps)", [follower.fee_bps, proxy.slippage_bps]], ["Requested latency (seconds)", follower.latency_seconds],
    ["Proxy mark age / execution wait limits (seconds)", [proxy.max_mark_age_seconds, proxy.max_wait_seconds]],
    ["Benchmark / split", [config.benchmark, config.split]], ["Dataset", data.dataset_id], ["Frozen configuration hash", data.config_hash],
  ];
  return <section className="panel"><h2>System card</h2><p className="muted">Frozen experiment settings, not current builder defaults. Leader PnL efficiency is not cash-flow-adjusted account ROI. Configured assets may have different historical eligibility start dates.</p><div className="table-scroll"><table><tbody>{fields.map(([key, value]) => <tr key={key}><th scope="row">{key}</th><td>{display(value)}</td></tr>)}</tbody></table></div>
    <details><summary>Complete frozen configuration & source identity</summary><pre>{JSON.stringify({ run_id: data.run_id, experiment_id: data.experiment_id, config: data.config, provenance: data.provenance }, null, 2)}</pre></details>
  </section>;
}

export function Diagnostics({ id, onTraders }: { id: string; onTraders: (date: string, instrument: string) => void }) {
  const state = useResource<DiagnosticData>(`/api/lab/experiments/${encodeURIComponent(id)}/diagnostics`);
  const data = state.data;
  return <div className="diagnostics"><Status {...state} />{data && <>
    <section className="panel"><p className="eyebrow">SAVED STRATEGY DIAGNOSTICS · READ ONLY</p><h2>{data.research_eligible === true ? "Research status recorded as eligible" : data.research_eligible === false ? "Not research-qualified" : "Research qualification unavailable"}</h2>
      <p>Descriptive development evidence only. Beating BTC while losing money does not establish profitability or risk-adjusted alpha.</p>
      <p className="notice">Original reconciliation: {data.reconciliation.accepted === false ? "not accepted" : data.reconciliation.accepted === true ? "accepted by original report" : "unavailable"}. Local arithmetic checks below cannot upgrade this status.</p>
      {data.warnings.map(w => <p className="notice" key={w}>{w.replace(/_/g, " ")}</p>)}
    </section>
    <div className="metric-cards">{["final_equity", "total_return", "sharpe", "sortino", "max_drawdown"].map(k => <article className="metric-card" key={k}><span>{label(k)}</span><MetricValue metric={data.stored_metrics[k]} /></article>)}</div>
    <SystemCard data={data} />
    <DiagnosticProfiles data={data} />
    <section className="panel"><h2>Stored risk & return metrics</h2><p className="muted">Saved sampling-frequency metrics; not recomputed from the display chart. See methodology for hourly/minute conventions.</p><MetricTable values={data.stored_metrics} benchmark={data.benchmark_metrics} /></section>
    <section className="panel"><h2>Weekly statistical analysis</h2><p className="muted">Complete Monday-to-Monday UTC weeks only. Partial periods are excluded. Weekly standard deviation and annualized volatility from weekly returns are separately labelled.</p><MetricTable values={Object.fromEntries(Object.entries(data.statistics).filter(([key]) => !key.includes("ci_")))} />
      <h3>Uncertainty · block-bootstrap 95% intervals</h3><p className="notice">Intervals describe uncertainty in the mean weekly return, not a forecast of next year's return. Four-week dependence blocks; one development window; no correction for strategy selection, missing history or proxy-model error.</p><MetricTable values={Object.fromEntries(Object.entries(data.statistics).filter(([key]) => key.includes("ci_")))} />
    </section>
    <DiagnosticAccounting data={data} />
    <DiagnosticCohorts data={data} onTraders={onTraders} />
    <section className="panel"><h2>Methodology & limitations</h2>{data.methods.map(m => <p className="muted" key={m}>{m}</p>)}</section>
  </>}</div>;
}
