import { useState } from "react";
import { candleEnd } from "../chartTime";
import type { Candle, HealthPoint, HealthSeries, Interval } from "../types";
import { datetime } from "../format";

function healthColor(point?: HealthPoint) {
  if (point?.status === "no-debt") return "hsl(140 72% 52%)";
  if (point?.status !== "available" || point.health == null || !Number.isFinite(point.health)) return "#33434e";
  const hue = Math.round(Math.max(0, Math.min(1, (point.health - 1) / .5)) * 140);
  return `hsl(${hue} 72% 52%)`;
}

export function HealthHeatmap({ candles, history, series, currentObligation, interval = "1d", plotLeft = 10, plotRight = 68 }: {
  candles: Candle[]; history: HealthPoint[]; series: HealthSeries[]; currentObligation?: string; interval?: Interval; plotLeft?: number; plotRight?: number;
}) {
  const [scope, setScope] = useState("current");
  const [hover, setHover] = useState<number | null>(null);
  const current = series.find(s => s.id === currentObligation);
  const points = scope === "all" ? history : scope === "current" ? current?.healthHistory || [] : series.find(s => s.id === scope)?.healthHistory || [];
  const byDay = new Map(points.map(p => [Math.floor(p.time / 86400) * 86400, p]));
  const cells = candles.flatMap((c,i) => {
    const end=candleEnd(c,interval);
    if(interval !== "1w" && interval !== "1M") return [{candle:c,point:byDay.get(Math.floor(c.time/86400)*86400),position:i,span:1}];
    const result=[];
    for(let time=c.time;time<end;time+=86400) result.push({candle:{...c,time},point:byDay.get(time),position:i+(time-c.time)/(end-c.time),span:86400/(end-c.time)});
    return result;
  });
  const active = hover !== null ? cells.find(c => c.candle.time === hover) : cells.at(-1);
  const point = active?.point;
  const label = point?.status === "no-debt" ? "No debt" : point?.status === "available" && point.health != null ? `${point.health.toFixed(3)}×` : "Historical health unavailable";
  const width = 900, left = plotLeft, right = plotRight, cellWidth = (width - left - right) / Math.max(candles.length, 1);
  return <div className="health-heatmap">
    <div className="health-heatmap-heading"><strong>Historical health</strong><select aria-label="Historical health obligation" value={scope} onChange={event => {setScope(event.target.value); setHover(null);}}><option value="current">Current obligation</option><option value="all">Worst across all obligations</option>{series.map(s => <option key={s.id} value={s.id}>{s.label}</option>)}</select></div>
    <svg viewBox={`0 0 ${width} 34`} role="img" aria-label="Historical Kamino health heatmap" tabIndex={0} onKeyDown={event => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key) || !cells.length) return;
      event.preventDefault();
      const index = hover === null ? cells.length - 1 : Math.max(0, cells.findIndex(c => c.candle.time === hover));
      const next = event.key === "Home" ? 0 : event.key === "End" ? cells.length - 1 : Math.max(0, Math.min(cells.length - 1, index + (event.key === "ArrowLeft" ? -1 : 1)));
      setHover(cells[next].candle.time);
    }} onPointerLeave={() => setHover(null)}>
      {cells.map(({candle, point, position, span}) => <rect key={candle.time} x={left + position * cellWidth} y="4" width={cellWidth * span + .1} height="24" fill={healthColor(point)} data-candle-time={candle.time} data-health-status={point?.status || "unavailable"} data-health-ratio={point?.health ?? undefined} tabIndex={-1} onPointerEnter={() => setHover(candle.time)} onFocus={() => setHover(candle.time)} onBlur={() => setHover(null)} aria-label={`${datetime(candle.time)}: ${point?.status === "no-debt" ? "no debt" : point?.status === "available" ? `health ${point.health?.toFixed(3)}` : "health unavailable"}`}>
        <title>{point?.status === "available" ? `Health ${point.health?.toFixed(3)}×` : point?.status === "no-debt" ? "No debt" : "No historical health data"} · {datetime(candle.time)} · provider daily bucket</title>
      </rect>)}
    </svg>
    <div className="health-heatmap-detail" aria-live="polite"><strong>{label}</strong><span>{active ? datetime(active.candle.time) : "No candles"}{point ? ` · Daily bucket ${datetime(point.time)}` : ""}</span>{point?.status === "available" && point.adjustedDebt != null && point.liquidationLimit != null && <span>Liquidation limit ${point.liquidationLimit.toLocaleString("en-US", {maximumFractionDigits: 2})} USD ÷ adjusted debt ${point.adjustedDebt.toLocaleString("en-US", {maximumFractionDigits: 2})} USD</span>}</div>
    <div className="health-heatmap-legend"><span><i style={{background:"hsl(0 72% 52%)"}}/> ≤1.00 liquidation threshold</span><span className="health-gradient"/><span><i style={{background:"hsl(140 72% 52%)"}}/> ≥1.50 / no debt</span><span><i style={{background:"#33434e"}}/> Missing data</span></div>
    <p className="muted">Daily Kamino snapshots · daily detail within weekly/monthly candles · gaps stay unknown. Ratio = liquidation limit ÷ borrow-factor-adjusted debt.</p>
  </div>;
}
