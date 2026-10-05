import { useEffect, useMemo, useState } from "react";
import { useChart } from "../api";
import { EquityOverlay } from "./EquityOverlay";
import { candleStart, candleEnd } from "../chartTime";
import { HealthHeatmap } from "./HealthHeatmap";
import type { Asset, Interval, Mode, Trade, HealthPoint, HealthSeries, Observation } from "../types";
import { date, datetime, money, number } from "../format";

const INTERVAL_LABELS: Record<Interval,string> = {"1h":"1H","4h":"4H","1d":"1 Day","1w":"1 Week","1M":"1 Month"};
function consensusColor(score: number | null | undefined) {
  if (score == null) return "#33434e";
  const t = Math.max(0,Math.min(1,(score+1)/2));
  const a=t<.5?[143,29,50]:[188,170,81], b=t<.5?[188,170,81]:[8,122,77], mix=t<.5?t*2:(t-.5)*2;
  return "#"+a.map((v,i)=>Math.round(v+(b[i]-v)*mix).toString(16).padStart(2,"0")).join("");
}
export function TradingChart({ mode, trades, selected, onSelect, asset, setAsset, days, end, start: inception, healthHistory = [], healthSeries = [], currentObligation, equityHistory = [], equityLabel = "Portfolio equity", equityDailyBuckets = false }: {
  mode: Mode; trades: Trade[]; selected: Trade | null; onSelect: (trade: Trade) => void;
  asset: Asset; setAsset: (asset: Asset) => void; days: number; end?: number; start?: number; healthHistory?: HealthPoint[]; healthSeries?: HealthSeries[]; currentObligation?: string; equityHistory?: Observation[]; equityLabel?: string; equityDailyBuckets?: boolean;
}) {
  const [interval, setInterval] = useState<Interval>("1d");
  const [period, setPeriod] = useState(10);
  const [multiplier, setMultiplier] = useState(3);
  const [showEquity, setShowEquity] = useState(false);
  const [combined, setCombined] = useState(false);
  const [showTrend, setShowTrend] = useState(true);
  const [showTrades, setShowTrades] = useState(true);
  const [settings, setSettings] = useState(false);
  const [activityBucket, setActivityBucket] = useState<number | null>(null);
  const [hover, setHover] = useState<number | null>(null);
  const [windowSize, setWindowSize] = useState(120);
  const [fitHistory, setFitHistory] = useState(false);
  const [offset, setOffset] = useState(0);
  const chart = useChart(mode, asset, interval, days, period, multiplier, end, inception, combined);
  const all = chart.data?.candles || [];
  const count = Math.min(fitHistory ? Math.min(all.length, 2500) : windowSize, all.length);
  const start = Math.max(0, all.length - count - Math.min(offset, all.length - count));
  const candles = all.slice(start, start + count);
  const candleIndices = new Map(candles.map((c, i) => [c.time, i]));
  const activityGroups = new Map<number, Trade[]>();
  for (const trade of trades) { const bucket = candleStart(trade.time, interval); if (candleIndices.has(bucket)) activityGroups.set(bucket, [...(activityGroups.get(bucket) || []), trade]); }
  const trend = (chart.data?.indicator || []).slice(start, start + count);
  useEffect(() => { setOffset(0); setHover(null); }, [asset, interval, days]);
  useEffect(() => { if (days === 0 && interval === "1d") { setFitHistory(true); setOffset(0); } }, [days, interval]);
  useEffect(() => {
    if (!selected || !all.length) return;
    const index = all.findIndex(candle => selected.time >= candle.time && selected.time < candleEnd(candle, interval));
    if (index >= 0) setOffset(Math.max(0, all.length - count - Math.max(0, index - Math.floor(count / 2))));
  }, [selected?.id, asset, chart.data, windowSize, fitHistory, interval]);
  const executions = useMemo(() => trades.flatMap(trade => {
    const leg = trade.executionLegs?.find(l => l.asset === asset);
    if (leg) return [{trade, side: leg.side, amount: leg.amount, price: leg.price, quote: leg.quote}];
    if (["buy", "sell"].includes(trade.type) && trade.asset === asset) return [{trade, side: trade.type, amount: trade.amount, price: trade.price, quote: trade.quote || "USD"}];
    return [];
  }), [trades, asset]);
  const w = 900, h = 378, left = showEquity ? 88 : 10, right = 68, top = 16, bottom = 276;
  const prices = candles.flatMap((c, i) => [c.low, c.high, ...(showTrend && trend[i]?.value != null ? [trend[i].value!] : [])]);
  const low = Math.min(...prices), high = Math.max(...prices), spread = (high - low) || 1;
  const x = (index: number) => left + (index + .5) / Math.max(count, 1) * (w - left - right);
  const y = (price: number) => bottom - 14 - (price - low) / spread * (bottom - top - 28);
  const barWidth = Math.max(1, Math.min(8, (w - left - right) / Math.max(count, 1) * .65));
  const activeIndex = hover === null ? candles.length - 1 : Math.min(candles.length - 1, hover);
  const active = candles[activeIndex];
  const activeTrend = trend[activeIndex];
  const consensus = (chart.data?.combined || []).slice(start,start+count);
  const activeConsensus = consensus[activeIndex];
  const trendSegments: { direction: string; points: string }[] = [];
  let continuingTrend = false;
  trend.forEach((point, i) => {
    if (point.value === null) { continuingTrend = false; return; }
    if (i > 0 && candleEnd(candles[i-1], interval) !== candles[i].time) continuingTrend = false;
    const last = trendSegments.at(-1);
    if (continuingTrend && last?.direction === point.direction) last.points += ` ${x(i)},${y(point.value)}`;
    else trendSegments.push({ direction: point.direction!, points: `${x(i)},${y(point.value)}` });
    continuingTrend = true;
  });
  const maxVolume = Math.max(1, ...candles.map(c => c.volume));
  return <section className="panel trading-chart">
    <div className="panel-heading"><div className="chart-title"><span className={`token token-${asset.toLowerCase()}`}>{asset === "SOL" ? "≋" : asset === "ETH" ? "◆" : "₿"}</span><div><h2>{asset} / USD <span className="market-tag">SPOT</span></h2><p className="muted">Price action & your executions</p></div></div><select aria-label="Chart asset" value={asset} onChange={e => setAsset(e.target.value as Asset)}>{["SOL", "ETH", "BTC"].map(a => <option key={a}>{a}</option>)}</select></div>
    <div className="chart-toolbar"><div className="segmented">{(["1d", "1w", "1M", "1h", "4h"] as Interval[]).map(value => <button key={value} className={interval === value ? "active" : ""} onClick={() => setInterval(value)}>{INTERVAL_LABELS[value]}</button>)}</div><div className="chart-toggles"><label className="equity-checkbox"><input type="checkbox" aria-label="Overlay equity" checked={showEquity} onChange={e=>setShowEquity(e.target.checked)}/>Equity</label><button className={combined ? "toggle active" : "toggle"} aria-pressed={combined} onClick={()=>setCombined(v=>!v)}>Combined Supertrend</button><button className={showTrend ? "toggle active" : "toggle"} onClick={() => setShowTrend(v => !v)} aria-pressed={showTrend}><i className="dot lime-bg"/>Supertrend</button><button className={showTrades ? "toggle active" : "toggle"} onClick={() => setShowTrades(v => !v)} aria-pressed={showTrades}><i className="dot cyan-bg"/>Transactions</button><button className="icon-button" onClick={() => setSettings(v => !v)} aria-expanded={settings} aria-label="Indicator settings">⚙</button></div></div>
    <p className="timeframe-caption muted">Candle interval: {interval === "1M" ? "1 calendar month" : interval === "1w" ? "1 week (Monday UTC)" : interval === "1d" ? "1 day" : interval === "4h" ? "4 hours" : "1 hour"} · closed candles · UTC{showEquity ? ` · Equity: ${equityLabel}` : ""}</p>
    {combined && <div className="combined-status">{(["1d","1w","1M"] as const).map((tf,i)=><span key={tf} className={activeConsensus?.directions[tf] === "bullish" ? "lime" : activeConsensus?.directions[tf] === "bearish" ? "negative" : "muted"}>{["Daily","Weekly","Monthly"][i]}: {activeConsensus?.directions[tf] || "warming up"}</span>)}<strong>{activeConsensus?.score == null ? "Combined warming up / missing data" : `${activeConsensus.bullishCount}/3 bullish · ${activeConsensus.bearishCount}/3 bearish`}</strong><span className="muted">Equal weights · states known at each candle close</span></div>}
    {combined && <div className="combined-legend">{[-1,-1/3,1/3,1].map((score,i)=><span key={score}><i className="dot" style={{background:consensusColor(score)}}/>{["3 bearish","2 bearish / 1 bullish","1 bearish / 2 bullish","3 bullish"][i]}</span>)}<span><i className="dot" style={{background:consensusColor(null)}}/>Unavailable</span></div>}
    {settings && <div className="indicator-settings"><label>ATR period<input type="number" aria-label="ATR period" min="1" max="100" value={period} onChange={e => setPeriod(Math.max(1, Math.min(100, Number(e.target.value) || 1)))}/></label><label>Multiplier<input type="number" aria-label="ATR multiplier" min="0.1" max="20" step="0.1" value={multiplier} onChange={e => setMultiplier(Math.max(.1, Math.min(20, Number(e.target.value) || .1)))}/></label><span className="muted">Wilder ATR · closed candles · UTC</span></div>}
    <div className="ohlc"><strong>{active ? money(active.close) : "—"}</strong>{active && <><span>O <b>{number(active.open)}</b></span><span>H <b>{number(active.high)}</b></span><span>L <b>{number(active.low)}</b></span><span>C <b className={active.close >= active.open ? "cyan" : "negative"}>{number(active.close)}</b></span></>}<span className={`trend-state ${activeTrend?.direction === "bullish" ? "lime" : "orange"}`}>{activeTrend?.direction || "warming up"}</span></div>
    {chart.loading ? <div className="empty-chart candle-empty" role="status">Loading {asset} candles…</div> : chart.error ? <div className="empty-chart candle-empty"><p className="negative">{chart.error}</p><span>Live charts never substitute demonstration candles.</span></div> : !candles.length ? <div className="empty-chart candle-empty">No candles available for this range.</div> : <svg className="candlestick-svg" viewBox={`0 0 ${w} ${h}`} role="img" aria-label={`${asset} candlestick chart with Supertrend and execution markers`} onPointerMove={e => {
      const bounds = e.currentTarget.getBoundingClientRect();
      const point = (e.clientX - bounds.left) / bounds.width * w;
      setHover(Math.max(0, Math.min(count - 1, Math.floor((point - left) / (w - left - right) * count))));
    }} onPointerLeave={() => setHover(null)}>
      {combined && candles.map((c,i)=><rect className="combined-cell" key={`consensus-${c.time}`} data-score={consensus[i]?.score ?? "unknown"} x={left+i/Math.max(count,1)*(w-left-right)} y={top} width={(w-left-right)/Math.max(count,1)+.1} height={bottom-top} fill={consensusColor(consensus[i]?.score)} opacity=".22"><title>{consensus[i]?.score == null ? "Combined Supertrend unavailable" : `${consensus[i].bullishCount}/3 bullish · ${consensus[i].bearishCount}/3 bearish`} · closed daily / weekly / monthly candles</title></rect>)}
      <text x={w-right+9} y="12" fill="#93a7b3" fontSize="10">Price USD</text>
      {[0, 1, 2, 3, 4].map(i => { const value = low + spread * i / 4; return <g key={i}><line x1={left} x2={w - right} y1={y(value)} y2={y(value)} stroke="#223039" strokeDasharray="2 5"/><text className="chart-price-tick" x={w - right + 9} y={y(value) + 4} fill="#73838d" fontSize="10">{number(value)}</text></g>; })}
      {candles.map((c, i) => { const color = combined && consensus[i]?.score != null ? consensusColor(consensus[i].score) : c.close >= c.open ? "#33d8c1" : "#ef8577"; return <g key={c.time} data-candle-time={c.time}><line x1={x(i)} x2={x(i)} y1={y(c.high)} y2={y(c.low)} stroke={color}/><rect x={x(i) - barWidth / 2} y={Math.min(y(c.open), y(c.close))} width={barWidth} height={Math.max(1, Math.abs(y(c.close) - y(c.open)))} fill={color}/><rect x={x(i) - barWidth / 2} y={322 - c.volume / maxVolume * 30} width={barWidth} height={c.volume / maxVolume * 30} fill={color} opacity=".25"/></g>; })}
      {showTrend && trendSegments.map((segment, i) => <polyline className="supertrend-line" key={i} points={segment.points} fill="none" stroke={segment.direction === "bullish" ? "#c7f45b" : "#ee9564"} strokeWidth="1.8"/>)}
      {showEquity && <EquityOverlay candles={candles} history={equityHistory} left={left} right={right} width={w} top={top} bottom={bottom-14} interval={interval} label={equityLabel} dailyBuckets={equityDailyBuckets}/>}
      <line x1={left} x2={w - right} y1={y(candles.at(-1)!.close)} y2={y(candles.at(-1)!.close)} stroke="#2ce4d9" strokeDasharray="3 4" opacity=".4"/>
      {showTrades && executions.map(({trade, side, amount, price, quote}) => {
        const i = candleIndices.get(candleStart(trade.time, interval)) ?? -1;
        if (i < 0) return null;
        const buy = side === "buy", cy = y(buy ? candles[i].low : candles[i].high) + (buy ? 15 : -15);
        return <g key={trade.id} role="button" tabIndex={0} aria-label={`${side} ${number(amount)} ${asset} on ${date(trade.time)} via ${trade.protocol}`} className="execution-marker" onClick={() => {onSelect(trade); setAsset(asset);}} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(trade); setAsset(asset); } }}>
          <circle cx={x(i)} cy={cy} r={selected?.id === trade.id ? 11 : 8} fill={buy ? "#c7f45b" : "#ee9564"} stroke="#0b1116" strokeWidth="2"/><path d={buy ? `M${x(i)-3},${cy+2} L${x(i)},${cy-2} L${x(i)+3},${cy+2}` : `M${x(i)-3},${cy-2} L${x(i)},${cy+2} L${x(i)+3},${cy-2}`} fill="none" stroke="#0b1116" strokeWidth="1.5"/><title>{`${side.toUpperCase()} ${number(amount)} ${asset} · ${number(price)} ${quote} per ${asset} · ${trade.protocol} · ${datetime(trade.time)}`}</title>
        </g>;
      })}
      {showTrades && [...activityGroups].map(([bucket, events]) => {
        const legs = events.filter(t => t.type === "liquidation");
        if (!legs.length) return null;
        const i = candleIndices.get(bucket)!;
        const count = new Set(legs.map(t => `${t.signature || t.id}:${t.obligation || ""}`)).size;
        return <g key={`liquidation-${bucket}`} role="button" tabIndex={0} className="execution-marker" aria-label={`${count} liquidation events on ${datetime(bucket)}`} onClick={() => {setActivityBucket(bucket); onSelect(legs[0]); setAsset(asset);}} onKeyDown={e => {if (e.key === "Enter" || e.key === " ") {e.preventDefault(); setActivityBucket(bucket); onSelect(legs[0]); setAsset(asset);}}}>
          <path d={`M${x(i)},338 l6,6 l-6,6 l-6,-6 Z`} fill="#f46d75" stroke="#0b1116"/><title>{count} liquidation events · {datetime(bucket)} · inspect seized collateral and repaid debt</title>
        </g>;
      })}
      <line x1={left} x2={w - right} y1="358" y2="358" stroke="#33434e"/>
      {showTrades && [...activityGroups].map(([bucket, events]) => {
        const i = candleIndices.get(bucket)!;
        const lending = events.some(t => ["kamino", "borrow", "repay", "deposit", "withdraw"].includes(t.type));
        return <g key={bucket} role="button" tabIndex={0} aria-label={`${events.length} wallet transactions on ${datetime(bucket)}: ${[...new Set(events.map(t => `${t.type} ${t.asset || t.protocol}`))].join(", ")}`} className="execution-marker" onClick={() => setActivityBucket(bucket)} onKeyDown={e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); setActivityBucket(bucket); } }}>
          <circle cx={x(i)} cy="358" r={activityBucket === bucket ? 7 : 5} fill={lending ? "#9b8cfa" : "#65b6d1"}/>
          {count <= 120 && <text x={x(i)} y="361" textAnchor="middle" fontSize="7" fill="#0b1116">{events.length}</text>}
          <title>{`${events.length} wallet transactions · ${datetime(bucket)} · click to inspect every action`}</title>
        </g>;
      })}
      <text x={left} y="374" fill="#93a7b3" fontSize="10">All wallet transactions · click a daily/timeframe group to inspect every action</text>
      {hover !== null && active && <><line x1={x(activeIndex)} x2={x(activeIndex)} y1={top} y2="325" stroke="#697e8a" strokeDasharray="3 4"/><circle cx={x(activeIndex)} cy={y(active.close)} r="3" fill="#d9e5eb"/></>}
      {[0, .25, .5, .75, 1].map(t => { const i = Math.round(t * (candles.length - 1)); return <text key={t} x={x(i)} y="337" textAnchor="middle" fill="#73838d" fontSize="10">{new Date(candles[i].time * 1000).toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric", timeZone: "UTC" })}</text>; })}
    </svg>}
    <HealthHeatmap candles={candles} history={healthHistory} series={healthSeries} currentObligation={currentObligation} interval={interval} plotLeft={left} plotRight={right}/>
    {activityBucket !== null && <div className="chart-activity-list"><div className="panel-heading"><strong>Transactions · {datetime(activityBucket)}</strong><button aria-label="Close chart transactions" onClick={() => setActivityBucket(null)}>×</button></div>{(activityGroups.get(activityBucket) || []).map(trade => <button key={trade.id} onClick={() => onSelect(trade)}><span>{datetime(trade.time)}</span><strong>{trade.type.toUpperCase()} {trade.asset || trade.protocol}</strong><span>{number(trade.amount)} {trade.asset || ""}</span></button>)}</div>}
    <div className="chart-footer"><span className="muted">{hover !== null && active ? datetime(active.time) : `${chart.data?.source || "Market data"} · ATR ${period} × ${multiplier}`}</span><div className="zoom-controls"><button aria-label="Zoom in" disabled={windowSize <= 20} onClick={() => { setFitHistory(false); setWindowSize(v => Math.max(20, Math.floor(v / 1.5))); }}>+</button><button aria-label="Zoom out" disabled={windowSize >= 600} onClick={() => setWindowSize(v => Math.min(600, Math.ceil(v * 1.5)))}>−</button><button onClick={() => { setFitHistory(false); setWindowSize(120); setOffset(0); }}>Reset</button><button onClick={() => { setFitHistory(true); setOffset(0); }}>Fit history</button></div></div>
    {all.length > count && <label className="pan-control"><span>Earlier candles</span><input type="range" aria-label="Chart history position" min="0" max={all.length - count} value={all.length - count - Math.min(offset, all.length - count)} onChange={e => setOffset(all.length - count - Number(e.target.value))}/><span>Latest</span></label>}
    <p className="muted chart-coverage">{candles.length} of {all.length} candles · {trades.filter(t => candles.length && t.time >= candles[0].time && t.time < candleEnd(candles.at(-1)!, interval)).length} of {trades.length} indexed transactions in view · UTC</p>
    <div className="chart-key"><span><i className="dot lime-bg"/> Buy / cover</span><span><i className="dot orange-bg"/> Sell / short</span><span><i className="dot" style={{background: "#9b8cfa"}}/> Kamino / lending</span><span><i className="dot" style={{background: "#65b6d1"}}/> Other activity</span><span><i className="dot" style={{background: "#f46d75"}}/> Liquidation</span><span className="muted">Click a marker to inspect the transaction</span></div>
  </section>;
}
