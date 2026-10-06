import { useEffect, useRef, useState } from "react";
import type { Observation, Metrics } from "../types";
import { money, number, percent } from "../format";

const fullDate = (time: number) => new Date(time * 1000).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" });
const utcTime = (time: number) => new Date(time * 1000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", timeZone: "UTC" });

export function Performance({ history, metrics, days, setDays, scopeLabel, dailyBuckets = false }: { history: Observation[]; metrics: Metrics; days: number; setDays: (days: number) => void; scopeLabel?: string; dailyBuckets?: boolean }) {
  const [unit, setUnit] = useState<"USD" | "SOL">("USD");
  const [hover, setHover] = useState<number | null>(null);
  const chartContainer = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(760);
  useEffect(() => {
    const container = chartContainer.current;
    if (!container) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(280, entry.contentRect.width)));
    observer.observe(container);
    return () => observer.disconnect();
  }, []);
  const anchor = history.at(-1)?.time || 0;
  const observations = history.filter(p => p.equity != null && Number.isFinite(p.equity) && (days === 0 || p.time >= anchor - days * 86400) && (unit === "USD" || (p.solPrice !== null && p.solPrice > 0)));
  const values = observations.map(p => unit === "USD" ? p.equity : p.equity / p.solPrice!);
  const height = 205, left = 76, right = width - 16, top = 24, bottom = 154;
  const min = values.length ? Math.min(...values) : 0, max = values.length ? Math.max(...values) : 0;
  const padding = (max - min) * .08 || Math.max(Math.abs(max) * .04, unit === "USD" ? 1 : .01);
  const low = min >= 0 ? Math.max(0, min - padding) : min - padding, high = max + padding;
  const start = observations[0]?.time || 0, end = observations.at(-1)?.time || start;
  const timeSpan = end - start;
  const timeX = (time: number) => left + (timeSpan ? (time - start) / timeSpan : .5) * (right - left);
  const x = (i: number) => timeX(observations[i].time);
  const y = (v: number) => bottom - (v - low) / (high - low) * (bottom - top);
  const line = values.map((v, i) => `${i && !((dailyBuckets && observations[i].time - observations[i-1].time > 1.5 * 86400) || observations[i].unpricedCount !== observations[i-1].unpricedCount || observations[i].valuationComplete !== observations[i-1].valuationComplete) ? "L" : "M"}${x(i)},${y(v)}`).join(" ");
  const index = hover !== null ? Math.min(hover, values.length - 1) : values.length - 1;
  const tickCount = width < 320 ? 2 : width < 650 ? 3 : 5;
  const axisValue = (value: number) => unit === "USD"
    ? value.toLocaleString("en-US", { style: "currency", currency: "USD", notation: Math.abs(value) >= 10000 ? "compact" : "standard", maximumFractionDigits: 2 })
    : value.toLocaleString("en-US", { notation: Math.abs(value) >= 10000 ? "compact" : "standard", maximumFractionDigits: 3 });
  return <section className="panel performance">
    <div className="panel-heading"><div><h2>{scopeLabel || "Portfolio performance"} <span className="muted mini">EQUITY</span></h2><p className="muted">{metrics.returnPct === null ? "Returns unavailable until cash flows are verified" : <><span className={metrics.returnPct >= 0 ? "lime" : "negative"}>{percent(metrics.returnPct, true)}</span> <span>flow-adjusted · selected range</span></>}</p></div>
      <div className="segmented range" aria-label="Performance range">{[7, 30, 90, 365, 0].map(d => <button key={d} className={days === d ? "active" : ""} aria-pressed={days === d} onClick={() => { setDays(d); setHover(null); }}>{d === 0 ? "Inception" : d === 365 ? "1Y" : `${d}D`}</button>)}</div></div>
    <div className="performance-value">{values.length ? unit === "USD" ? money(values[index]) : `${number(values[index], 3)} SOL` : "—"}<div className="segmented"><button className={unit === "USD" ? "active" : ""} onClick={() => { setUnit("USD"); setHover(null); }}>USD</button><button className={unit === "SOL" ? "active" : ""} onClick={() => { setUnit("SOL"); setHover(null); }}>SOL</button></div></div>
    <div className="equity-chart-container" ref={chartContainer}>{values.length > 1 ? <svg className="performance-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Portfolio equity history" onPointerMove={event => {
      const point = event.currentTarget.createSVGPoint();
      point.x = event.clientX; point.y = event.clientY;
      const transform = event.currentTarget.getScreenCTM();
      if (!transform) return;
      const position = point.matrixTransform(transform.inverse()).x;
      const time = start + Math.max(0, Math.min(1, (position - left) / (right - left))) * timeSpan;
      setHover(observations.reduce((nearest, p, i) => Math.abs(p.time - time) < Math.abs(observations[nearest].time - time) ? i : nearest, 0));
    }} onPointerLeave={() => setHover(null)}>
      <defs><linearGradient id="equity-fill" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#2ce4d9" stopOpacity=".18"/><stop offset="1" stopColor="#2ce4d9" stopOpacity="0"/></linearGradient></defs>
      <text className="equity-axis-unit" x={left - 10} y="13" textAnchor="end">{unit}</text>
      {[0, .25, .5, .75, 1].map(t => { const value = low + t * (high - low); return <g key={t}><line x1={left} y1={y(value)} x2={right} y2={y(value)} stroke="#233139" strokeDasharray="3 6"/><text className="equity-y-tick" x={left - 10} y={y(value)} dominantBaseline="middle" textAnchor="end">{axisValue(value)}</text></g>; })}
      {!dailyBuckets && observations.every(p => p.valuationComplete !== false) && <path d={`${line} L${x(values.length - 1)},${bottom} L${x(0)},${bottom}Z`} fill="url(#equity-fill)"/>}<path className="equity-line" d={line} fill="none" stroke="#2ce4d9" strokeWidth="2"/>
      {Array.from({ length: tickCount }, (_, i) => { const time = start + i / (tickCount - 1) * timeSpan; return <g key={i}><line x1={timeX(time)} x2={timeX(time)} y1={bottom} y2={bottom + 5} stroke="#50606b"/><text className="equity-x-tick" x={timeX(time)} y={bottom + 21} textAnchor={i === 0 ? "start" : i === tickCount - 1 ? "end" : "middle"}>{fullDate(time)}</text>{timeSpan < 2 * 86400 && <text className="equity-time-tick" x={timeX(time)} y={bottom + 37} textAnchor={i === 0 ? "start" : i === tickCount - 1 ? "end" : "middle"}>{utcTime(time)} UTC</text>}</g>; })}
      {index >= 0 && <><line x1={x(index)} x2={x(index)} y1={top} y2={bottom} stroke="#50606b" strokeDasharray="3 4"/><circle cx={x(index)} cy={y(values[index])} r="4" fill="#2ce4d9" stroke="#0c1419" strokeWidth="2"/></>}
    </svg> : <div className="empty-chart">{observations.length === 1 ? "One equity observation in this range. Select Inception to check all available history." : "No historical equity observations available."}</div>}</div>
    {observations.some(p => p.valuationComplete === false) && <p className="muted equity-scope-note">Priced portion only · unsupported tokens excluded. Coverage changes break the line.</p>}
    {dailyBuckets && <p className="muted equity-scope-note">Kamino collateral minus debt · wallet assets excluded · provider daily buckets · gaps are not interpolated.</p>}
    <div className="chart-dates"><span>{hover !== null && observations[index] ? `${fullDate(observations[index].time)} · ${utcTime(observations[index].time)} UTC` : days === 0 ? "All available equity observations · UTC" : "Account value includes external flows · UTC"}</span></div>
  </section>;
}
