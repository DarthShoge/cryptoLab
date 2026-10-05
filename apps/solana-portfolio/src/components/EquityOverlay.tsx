import type { Candle, Interval, Observation } from "../types";
import { candleEnd } from "../chartTime";

type Props = {
  candles: Candle[];
  history: Observation[];
  left: number;
  right: number;
  width: number;
  top: number;
  bottom: number;
  interval: Interval;
  label: string;
  dailyBuckets?: boolean;
};

type LocatedPoint = { observation: Observation; x: number };
const PURPLE = "#b69bff";

function locate(time: number, candles: Candle[], interval: Interval): number {
  let low = 0, high = candles.length - 1, index = -1;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (candles[middle].time <= time) { index = middle; low = middle + 1; }
    else high = middle - 1;
  }
  return index >= 0 && time < candleEnd(candles[index], interval) ? index : -1;
}

function coverage(point: Observation): string {
  return `${point.valuationComplete ?? "unknown"}:${point.unpricedCount ?? "unknown"}`;
}

function dollars(value: number): string {
  return Math.abs(value) >= 1e12 ? `$${value.toExponential(2)}` : new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", notation: "compact", maximumFractionDigits: 1,
  }).format(value);
}

export function EquityOverlay({ candles, history, left, right, width, top, bottom, interval, label, dailyBuckets = false }: Props) {
  const segments: LocatedPoint[][] = [];
  let segment: LocatedPoint[] = [];
  let previous: Observation | null = null;
  const flush = () => { if (segment.length) segments.push(segment); segment = []; previous = null; };
  const plotWidth = width - left - right;
  for (const observation of [...history].sort((a, b) => a.time - b.time)) {
    if (!Number.isFinite(observation.time) || observation.equity == null || !Number.isFinite(observation.equity)) {
      flush();
      continue;
    }
    const index = locate(observation.time, candles, interval);
    if (index < 0) { flush(); continue; }
    if (previous && (coverage(previous) !== coverage(observation) ||
      (dailyBuckets && observation.time - previous.time > 1.5 * 86400))) flush();
    const candle = candles[index];
    const duration = candleEnd(candle, interval) - candle.time;
    if (!(duration > 0)) { flush(); continue; }
    const fraction = (observation.time - candle.time) / duration;
    segment.push({ observation, x: left + (index + fraction) / candles.length * plotWidth });
    previous = observation;
  }
  flush();
  const values = segments.flatMap(points => points.map(point => point.observation.equity));
  const hasEquity = values.length > 0;
  // Normalize the scale first so even very large finite inputs cannot overflow
  // the difference between positive and negative equity.
  const unit = hasEquity ? Math.max(1, ...values.map(value => Math.abs(value))) : 1;
  const normalized = values.map(value => value / unit);
  const minimum = hasEquity ? Math.min(...normalized) : 0;
  const maximum = hasEquity ? Math.max(...normalized) : 1;
  const padding = maximum === minimum ? Math.max(.01, Math.abs(maximum) * .03) : (maximum - minimum) * .05;
  const low = minimum - padding, high = maximum + padding;
  const y = (equity: number) => bottom - (equity / unit - low) / (high - low) * (bottom - top);
  const source = `${label}${dailyBuckets ? " · provider daily bucket labels" : " · observed equity"}`;
  const title = (point: LocatedPoint) => `${source} · ${new Date(point.observation.time * 1000).toISOString()} · ${dollars(point.observation.equity)}`;
  return <g className="chart-equity-axis" data-no-equity={hasEquity ? undefined : "true"} role="group" aria-label={`${source} · independent USD equity axis`}>
    <title>{`${source} · equity values, independent of the market price axis. Missing values and coverage changes are not connected.`}</title>
    <text x={left - 8} y={top - 4} textAnchor="end" fill={PURPLE} fontSize="10">Equity USD</text>
    {hasEquity ? <>
      <line x1={left} x2={left} y1={top} y2={bottom} stroke={PURPLE} opacity=".35"/>
      {[0, .25, .5, .75, 1].map(fraction => {
        const normalizedValue = low + (high - low) * fraction;
        const value = Math.sign(normalizedValue) * Math.min(Number.MAX_VALUE, Math.abs(normalizedValue) * unit);
        const tickY = bottom - fraction * (bottom - top);
        return <g key={fraction}><line x1={left - 4} x2={left} y1={tickY} y2={tickY} stroke={PURPLE}/><text x={left - 8} y={tickY + 3} textAnchor="end" fill={PURPLE} fontSize="10">{dollars(value)}</text></g>;
      })}
      {segments.map((points, index) => points.length === 1
        ? <circle key={index} className="chart-equity-line" cx={points[0].x} cy={y(points[0].observation.equity)} r="3" fill={PURPLE} aria-label={title(points[0])}><title>{title(points[0])}</title></circle>
        : <polyline key={index} className="chart-equity-line" points={points.map(point => `${point.x},${y(point.observation.equity)}`).join(" ")} fill="none" stroke={PURPLE} strokeWidth="2" aria-label={`${source} · ${points.length} observations`}><title>{`${title(points[0])} through ${title(points[points.length - 1])}`}</title></polyline>)}
    </> : <text x={left - 8} y={top + 12} textAnchor="end" fill="#93a7b3" fontSize="10">Unavailable</text>}
  </g>;
}
