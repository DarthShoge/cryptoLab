import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Plot } from "./Charts";
import { formatValue, utc } from "./format";
import type { DiagnosticData } from "./DiagnosticTables";

export function DiagnosticProfiles({ data }: { data: DiagnosticData }) {
  const strategy = data.series.strategy;
  const base = strategy.points ?? [];
  const points = new Map<number, Record<string, number | null>>();
  const lines: [string, string, string][] = [];
  for (const [key, name, color] of [["strategy", "Strategy", "#7fe2ca"], ["btc", "BTC perpetual", "#8b9de8"], ["cash", "Cash", "#d9bd7c"]]) {
    const series = data.series[key]?.points ?? [];
    if (!base.length || series[0]?.time !== base[0].time || series.at(-1)?.time !== base.at(-1)?.time) continue;
    lines.push([key, name, color]);
    for (const p of series) {
      const t = Date.parse(p.time);
      points.set(t, { ...points.get(t), timestamp: t, [key]: p.growth, [`${key}Drawdown`]: p.drawdown,
        ...(key === "strategy" ? { gross: p.gross_exposure ?? null, net: p.net_exposure ?? null } : {}) });
    }
  }
  const chartRows = [...points.values()].sort((a, b) => a.timestamp! - b.timestamp!);
  const weeks = strategy.weeks ?? [];
  const complete = weeks.filter(w => !w.partial);
  const sorted = [...complete].sort((a, b) => a.return_value - b.return_value);
  const min = sorted[0]?.return_value ?? 0, max = sorted.at(-1)?.return_value ?? 0;
  const width = max > min ? (max - min) / 10 : .01;
  const bins = Array.from({ length: 10 }, (_, i) => ({ bucket: `${formatValue(min + width * i, "percent")} to ${formatValue(min + width * (i + 1), "percent")}`, count: 0 }));
  complete.forEach(w => { bins[Math.min(9, Math.floor((w.return_value - min) / width))].count++; });
  function periods(kind: "weeks" | "months") {
    const other = new Map((data.series.btc[kind] ?? []).map(w => [`${w.start}/${w.end}`, w.return_value]));
    return <div className="table-scroll"><table><thead><tr><th>Start (UTC)</th><th>End (UTC)</th><th>Period</th><th>Strategy</th><th>BTC perpetual</th></tr></thead><tbody>{(strategy[kind] ?? []).map(w => <tr key={w.start}><td>{utc(w.start)}</td><td>{utc(w.end)}</td><td>{w.partial ? "Partial · excluded from weekly inference" : "Complete"}</td><td>{formatValue(w.return_value, "percent")}</td><td>{formatValue(other.get(`${w.start}/${w.end}`), "percent")}</td></tr>)}</tbody></table></div>;
  }
  return <section><h2>Return profiles</h2>
    <p className="muted">{strategy.samples.toLocaleString()} validated source observations · UTC · daily extrema-preserving display. Growth starts at 1. Only controls with identical full-window endpoints are plotted; BTC is a funded perpetual control, not spot BTC.</p>
    {Object.entries(data.series).map(([key, series]) => series.reason && <p className="notice" key={key}>{key}: {series.reason}</p>)}
    {base.length > 0 && <>
      <Plot title="Growth of initial capital" rows={chartRows} unit="ratio" lines={lines} />
      <div className="chart-grid"><Plot title="Drawdown from running peak" rows={chartRows} unit="percent" lines={lines.map(([k, n, c]) => [`${k}Drawdown`, n, c])} /><Plot title="Gross & net exposure (USD)" rows={chartRows} unit="usd" lines={[["gross", "Gross", "#8b9de8"], ["net", "Net", "#7fe2ca"]]} /></div>
      <details className="panel"><summary>Accessible chart data · displayed observations</summary><div className="table-scroll" style={{ maxHeight: 360 }}><table><thead><tr><th>Time (UTC)</th>{lines.flatMap(([k, n]) => [<th key={k}>{n} growth</th>, <th key={`${k}DD`}>{n} drawdown</th>])}<th>Gross USD</th><th>Net USD</th></tr></thead><tbody>{chartRows.map(r => <tr key={r.timestamp}><td>{utc(new Date(r.timestamp!).toISOString())}</td>{lines.flatMap(([k]) => [<td key={k}>{formatValue(r[k], "ratio")}</td>, <td key={`${k}DD`}>{formatValue(r[`${k}Drawdown`], "percent")}</td>])}<td>{formatValue(r.gross, "usd")}</td><td>{formatValue(r.net, "usd")}</td></tr>)}</tbody></table></div></details>
    </>}
    <section className="panel"><h3>Calendar-month returns</h3>{periods("months")}</section>
    <section className="panel"><h3>Complete-week return distribution</h3><p className="muted">{complete.length} complete weeks; {weeks.length - complete.length} partial periods excluded. Positive-return weeks are wins, not individual trades.</p>
      {complete.length ? <><div className="chart" role="img" aria-label="Complete-week return histogram; counts in table below"><ResponsiveContainer width="100%" height="100%"><BarChart data={bins}><CartesianGrid stroke="#273540" /><XAxis dataKey="bucket" tick={false} /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count" fill="#7fe2ca" isAnimationActive={false} /></BarChart></ResponsiveContainer></div><details><summary>Histogram counts</summary><table><thead><tr><th>Return bin (lower inclusive; final upper inclusive)</th><th>Weeks</th></tr></thead><tbody>{bins.map(b => <tr key={b.bucket}><td>{b.bucket}</td><td>{b.count}</td></tr>)}</tbody></table></details></> : <p>No complete weeks available.</p>}
      <details><summary>All weekly returns, including partial periods</summary>{periods("weeks")}</details>
    </section>
  </section>;
}
