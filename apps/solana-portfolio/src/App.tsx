import { useEffect, useRef, useState } from "react";
import { usePortfolio } from "./api";
import type { Asset, Metrics, Portfolio, Trade, View } from "./types";
import { datetime, money, number, percent, short } from "./format";
import { Icon } from "./components/Icon";
import { Performance } from "./components/Performance";
import { TradingChart } from "./components/TradingChart";
import { LoanHealth } from "./components/LoanHealth";
import { Positions } from "./components/Positions";
import { Activity, TradeDetail } from "./components/Activity";

const NAV = [{ id: "overview", label: "Portfolio", icon: "overview" }, { id: "analytics", label: "Trading analytics", icon: "chart" }, { id: "positions", label: "Positions", icon: "positions" }, { id: "activity", label: "Activity", icon: "activity" }] as const;
const TITLES: Record<View, string> = { overview: "Portfolio overview", analytics: "Trading analytics", positions: "Positions & exposure", activity: "Account activity" };
const EMPTY_METRICS: Metrics = { pnl: null, returnPct: null, solPnl: null, drawdownPct: null };

function SummaryCards({ data, metrics, days }: { data: Portfolio; metrics: Metrics; days: number }) {
  const summary = data.summary;
  const sol = data.prices.SOL;
  return <div className="summary-cards">
    <article className="metric-card equity-card"><div className="metric-label"><i className="dot cyan-bg"/> {summary.netEquity === null ? "Known portfolio equity" : "Net portfolio equity"} <span className="mini muted">USD</span></div><div className="metric-value">{money(summary.netEquity ?? (summary.positions.length ? summary.knownEquity : null))}</div><div className="metric-bottom"><span className="muted">{summary.netEquity !== null && sol ? `${number(summary.netEquity / sol, 3)} SOL equivalent` : summary.unpricedCount ? `Incomplete · ${summary.unpricedCount} unpriced tokens` : "Incomplete provider coverage"}</span><Icon name="wallet" size={18}/></div></article>
    <article className="metric-card"><div className="metric-label">Flow-adjusted P&L <span className="mini range-pill">{days === 0 ? "Inception" : days === 365 ? "1Y" : `${days}D`}</span></div><div className={`metric-value smaller ${metrics.pnl !== null && metrics.pnl < 0 ? "negative" : "cyan"}`}>{money(metrics.pnl)}</div><div className="metric-bottom"><span className="muted">{metrics.pnl === null ? "Needs equity & cash-flow coverage" : data.mode === "imported" ? "User-declared complete history" : "External deposits excluded"}</span><span className={metrics.returnPct !== null && metrics.returnPct < 0 ? "negative" : "lime"}>{percent(metrics.returnPct, true)}</span></div></article>
    <article className="metric-card"><div className="metric-label">Supplied capital <span className="mini muted">KAMINO</span></div><div className="metric-value smaller">{money(summary.supplied)}</div><div className="metric-bottom"><span className="muted">{summary.loans.length} obligation{summary.loans.length === 1 ? "" : "s"}</span><span className="lime">Collateral</span></div></article>
    <article className="metric-card debt-card"><div className="metric-label">Outstanding debt <span className="mini muted">MARK TO MARKET</span></div><div className="metric-value smaller orange">{money(summary.debt)}</div><div className="metric-bottom"><span className="muted">Stablecoin loans + token shorts</span><Icon name="positions" size={18}/></div></article>
  </div>;
}

function Allocation({ data }: { data: Portfolio }) {
  const holdings = data.summary.positions.filter(p => p.kind !== "borrowed" && p.value != null);
  const totals = new Map<string, number>();
  holdings.forEach(p => totals.set(p.symbol, (totals.get(p.symbol) || 0) + p.value!));
  const assets = [...totals].sort((a, b) => b[1] - a[1]);
  const total = assets.reduce((s, a) => s + a[1], 0);
  const palette = ["#2ce4d9", "#c7f45b", "#9380ff", "#ee9564", "#6395fa"];
  let cursor = 0;
  const stops = assets.map(([, value], i) => { const start = cursor; cursor += value / total * 100; return `${palette[i % palette.length]} ${start}% ${cursor}%`; }).join(",");
  return <section className="panel allocation-panel"><div className="panel-heading"><h2>Asset allocation</h2><span className="muted mini">GROSS ASSETS</span></div><div className="allocation-content"><div className="donut" style={{ background: total ? `conic-gradient(${stops})` : "#1b2931" }} role="img" aria-label="Gross asset allocation"><div><strong>{assets.length}</strong><span>assets</span></div></div><div className="allocation-legend">{assets.length ? assets.slice(0, 5).map(([symbol, value], i) => <div key={symbol}><span><i className="dot" style={{ background: palette[i % palette.length] }}/>{symbol}</span><strong>{percent(value / total * 100)}</strong></div>) : <p className="muted">No priced holdings loaded</p>}</div></div><p className="allocation-note muted">Wallet + supplied assets · debt shown separately</p></section>;
}

function AnalyticsStats({ data, metrics }: { data: Portfolio; metrics: Metrics }) {
  const executions = data.trades.filter(t => ["buy", "sell"].includes(t.type));
  const buys = executions.filter(t => t.type === "buy").length;
  return <div className="analytics-stats"><div><span>Observed DEX executions</span><strong>{executions.length}</strong><small>{buys} buys · {executions.length - buys} sells</small></div><div><span>Maximum drawdown</span><strong className="orange">{percent(metrics.drawdownPct)}</strong><small>On flow-adjusted returns</small></div><div><span>SOL-equivalent P&L</span><strong className="cyan">{number(metrics.solPnl, 3)} {metrics.solPnl === null ? "" : "SOL"}</strong><small>External flows removed at their SOL price</small></div><div><span>Realized trade P&L</span><strong>—</strong><small>Requires complete opening cost basis</small></div></div>;
}

export default function App() {
  const portfolio = usePortfolio();
  const [view, setView] = useState<View>("overview");
  const [days, setDays] = useState(90);
  const [equityScope, setEquityScope] = useState("current-kamino");
  const [asset, setAsset] = useState<Asset>("SOL");
  const [selected, setSelected] = useState<Trade | null>(null);
  const [settings, setSettings] = useState(false);
  const [walletInput, setWalletInput] = useState("");
  const [importError, setImportError] = useState("");
  const [importing, setImporting] = useState(false);
  const closeSettings = useRef<HTMLButtonElement>(null);
  const data = portfolio.data;
  const historicalKamino = data?.mode === "live" && equityScope !== "portfolio";
  const performanceHistory = historicalKamino ? (equityScope === "kamino" ? data?.kaminoHistory : equityScope === "current-kamino" ? data?.kaminoSeries?.find(s => s.id === data.summary.loans[0]?.address)?.history : data?.kaminoSeries?.find(s => s.id === equityScope)?.history) || [] : data?.history || [];
  const metrics = (days === 0 ? data?.metrics : data?.metricsByRange[String(days)]) || EMPTY_METRICS;
  const selectTrade = (trade: Trade) => { setSelected(trade); if (days && data && trade.time < data.asOf - days * 86400) setDays(0); if (["SOL", "ETH", "BTC"].includes(trade.asset || "")) setAsset(trade.asset as Asset); };
  useEffect(() => { setSelected(null); setDays(portfolio.mode === "demo" ? 90 : 0); }, [portfolio.mode, portfolio.wallet]);
  useEffect(() => { if (settings) { setWalletInput(portfolio.wallet); closeSettings.current?.focus(); } }, [settings]);
  useEffect(() => {
    if (!settings) return;
    const previous = document.activeElement as HTMLElement | null;
    const close = (event: KeyboardEvent) => {
      if (event.key === "Escape") setSettings(false);
      if (event.key === "Tab") {
        const nodes = [...document.querySelectorAll<HTMLElement>('.settings-modal button, .settings-modal input, .settings-modal a[href]')].filter(node => !node.hasAttribute("disabled"));
        const first = nodes[0], last = nodes.at(-1);
        if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
        else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
      }
    };
    document.addEventListener("keydown", close);
    return () => { document.removeEventListener("keydown", close); previous?.focus(); };
  }, [settings]);
  const openSettings = () => { setImportError(""); setSettings(true); };
  const sampleHistory = () => {
    const sample = { source: { name: "Your historical export", account: "Your wallet label", equityScope: "wallet-and-kamino", cashFlowCoverage: "complete" }, complete: true, flowTiming: "period-end", history: [{ time: 1790553600, equity: 1000, externalFlow: 0, solPrice: 118 }, { time: 1790640000, equity: 1600, externalFlow: 500, solPrice: 120 }], trades: [] };
    const url = URL.createObjectURL(new Blob([JSON.stringify(sample, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "portfolio-history-example.json"; link.click(); URL.revokeObjectURL(url);
  };
  return <div className="app-shell">
    <aside className="sidebar"><a className="brand" href="#" onClick={e => { e.preventDefault(); setView("overview"); }}><span className="brand-icon"><Icon name="orbit" size={32}/></span><span><strong>Orbit<span className="brand-dot">.</span></strong><small>SOLANA PORTFOLIO</small></span></a><div className="nav-label">YOUR WORKSPACE</div><nav aria-label="Main navigation">{NAV.map(item => <button key={item.id} className={view === item.id ? "nav-item active" : "nav-item"} onClick={() => setView(item.id)} aria-current={view === item.id ? "page" : undefined}><Icon name={item.icon}/><span>{item.label}</span>{view === item.id && <i className="dot cyan-bg"/>}</button>)}</nav>
      <div className="sidebar-market"><div className="nav-label">MARKET PULSE <span className="mini">USD</span></div>{(["SOL", "ETH", "BTC"] as Asset[]).map(symbol => <button key={symbol} className="pulse-row" onClick={() => { setAsset(symbol); setView("analytics"); }}><span>{symbol}</span><strong>{money(data?.prices[symbol], symbol === "BTC" ? 0 : 2)}</strong></button>)}<span className="pulse-source muted">{portfolio.mode === "demo" ? "Demonstration prices" : "Supported provider marks"}</span></div>
      <div className="sidebar-bottom"><button className="nav-item" onClick={openSettings}><Icon name="settings"/><span>Data & settings</span></button><div className="connection"><i className={`dot ${portfolio.mode === "live" ? "cyan-bg" : "orange-bg"}`}/><span>{portfolio.mode === "live" ? "SOLANA MAINNET" : portfolio.mode === "imported" ? "IMPORTED HISTORY" : "DEMO WORKSPACE"}</span><small>Read-only portfolio analytics</small></div></div>
    </aside>
    <main><header className="page-header"><div><div className="breadcrumb">PERSONAL VAULT <span>/</span><b>{view === "overview" ? "Portfolio" : view.charAt(0).toUpperCase() + view.slice(1)}</b></div><h1>{TITLES[view]}</h1><p>One view of your capital, your leverage, and your decisions.</p></div><div className="header-actions"><span className="network"><i className="dot cyan-bg"/>Solana</span><button className="wallet-button" onClick={openSettings}><Icon name="wallet"/>{data?.wallet ? short(data.wallet) : portfolio.mode === "demo" ? "Demo account" : "Account settings"}<span>⌄</span></button></div></header>
      <section className="market-prices" aria-label="Market prices">
        <div className="market-price-values">{(["ETH", "SOL", "BTC"] as Asset[]).map(symbol => <button key={symbol} className="market-price" onClick={() => {setAsset(symbol); setView("analytics");}} aria-label={`View ${symbol} chart`}><span>{symbol} <small>USD</small></span><strong>{money(data?.prices[symbol], symbol === "BTC" ? 0 : 2)}</strong></button>)}</div>
        <div className="market-price-meta"><span className="muted">{portfolio.mode === "demo" ? "Demonstration prices" : portfolio.mode === "imported" ? "Imported price marks" : "Provider spot prices"}{data ? ` · Updated ${datetime(data.asOf)}` : ""}</span><button className="refresh-button" disabled={portfolio.loading || portfolio.mode === "imported"} onClick={portfolio.refresh} aria-label="Refresh account" title={portfolio.mode === "imported" ? "Imported history is a fixed snapshot" : "Refresh account, prices and charts"}><Icon name="refresh" size={15}/>{portfolio.loading ? "Refreshing…" : "Refresh"}</button></div>
      </section>
      {portfolio.mode === "demo" && <div className="mode-banner"><span><i className="dot orange-bg"/><strong>Fictional demonstration</strong><span>Preview both lending & trading workflows.</span></span><button onClick={openSettings}>Load my wallet <Icon name="arrow" size={12}/></button></div>}
      {portfolio.mode === "imported" && <div className="mode-banner imported-banner"><span><i className="dot cyan-bg"/><strong>Imported history</strong><span>Coverage declared by the uploaded file.</span></span><button onClick={() => portfolio.setMode("demo")}>Return to demo</button></div>}
      {portfolio.loading ? <section className="loading-state" role="status"><span className="loading-orbit"><Icon name="orbit" size={46}/></span><h2>{portfolio.mode === "live" ? "Reading your Solana account" : "Preparing your portfolio"}</h2><p className="muted">{portfolio.mode === "live" ? "Loading Kamino obligations, wallet balances and recent activity. RPC history can take a minute." : "Loading equity, health and trade history…"}</p></section> : portfolio.error ? <section className="panel error-state" role="alert"><h2>Account data unavailable</h2><p>{portfolio.error}</p><button className="primary-button" onClick={portfolio.refresh}>Try again</button><button onClick={openSettings}>Review data settings</button></section> : data && <>
        <SummaryCards data={data} metrics={metrics} days={days}/>
        {data.historyStatus && <p className="history-progress" role="status">{data.historyStatus.discoveryComplete && data.historyStatus.decodingComplete ? "Wallet history indexed" : "Backfilling wallet history"} · {data.historyStatus.processed.toLocaleString()} / {data.historyStatus.discovered.toLocaleString()} transactions processed{data.historyStatus.oldest ? ` · inception ${new Date(data.historyStatus.oldest * 1000).toLocaleDateString("en-GB", {timeZone: "UTC"})}` : ""}{data.historyStatus.failed ? ` · ${data.historyStatus.failed.toLocaleString()} failed attempts` : ""}{data.historyStatus.missing ? ` · ${data.historyStatus.missing} unavailable, retrying` : ""}{data.historyStatus.error ? ` · ${data.historyStatus.error}` : ""}. Chart markers update as records arrive.</p>}
        {data.warnings.length > 0 && <details className="coverage-notice"><summary><i className="dot orange-bg"/>Data coverage · {data.warnings.length} notes{data.summary.netEquity === null ? " · valuation incomplete" : ""}</summary><ul>{data.warnings.map(note => <li key={note}>{note}</li>)}</ul></details>}
        {(view === "overview" || view === "analytics") && <>
          {data.mode === "live" && <div className="equity-scope-control"><label>Equity history <select aria-label="Equity history scope" value={equityScope} onChange={e => setEquityScope(e.target.value)}><option value="portfolio">Portfolio observations</option><option value="current-kamino">Current Kamino obligation</option><option value="kamino">All Kamino obligations</option>{data.kaminoSeries?.map(series => <option key={series.id} value={series.id}>{series.label}</option>)}</select></label>{historicalKamino && <span className="muted">Historical Kamino values exclude assets held in your wallet.</span>}</div>}
          {historicalKamino && !!data.kaminoHistoryWarnings?.length && <details className="coverage-notice"><summary>Historical Kamino coverage · {data.kaminoHistoryWarnings.length} notes</summary><ul>{data.kaminoHistoryWarnings.map(note => <li key={note}>{note}</li>)}</ul></details>}
          <div className="overview-top"><Performance history={performanceHistory} metrics={historicalKamino ? EMPTY_METRICS : metrics} days={days} setDays={setDays} scopeLabel={historicalKamino ? "Kamino net equity" : data.mode === "live" ? "Known portfolio equity" : undefined} dailyBuckets={historicalKamino}/><Allocation data={data}/></div>
          {view === "analytics" && <AnalyticsStats data={data} metrics={metrics}/>}
          <div className="trading-grid"><TradingChart mode={portfolio.mode} trades={data.trades} selected={selected} onSelect={selectTrade} asset={asset} setAsset={setAsset} days={days} start={data.historyStatus?.oldest || data.history[0]?.time || Math.min(...data.trades.map(t => t.time))} end={data.mode === "imported" ? data.asOf : undefined} equityHistory={performanceHistory} equityLabel={historicalKamino ? "Kamino net equity · wallet excluded" : data.mode === "live" ? "Known portfolio equity · incomplete valuation" : "Portfolio equity"} equityDailyBuckets={historicalKamino} healthHistory={data.healthHistory} healthSeries={data.kaminoSeries} currentObligation={data.summary.loans[0]?.address}/><LoanHealth data={data}/></div>
        </>}
        {selected && <TradeDetail trade={selected} relatedTrades={data.trades} onClose={() => setSelected(null)}/>}
        {view === "overview" && <div className="bottom-grid"><Positions data={data} onViewAll={() => setView("positions")}/><Activity trades={data.trades} selected={selected} onSelect={selectTrade} onViewAll={() => setView("activity")}/></div>}
        {view === "analytics" && <Activity trades={data.trades} selected={selected} onSelect={selectTrade} expanded/>}
        {view === "positions" && <div className="positions-view"><Positions data={data} expanded/><LoanHealth data={data}/></div>}
        {view === "activity" && <Activity trades={data.trades} selected={selected} onSelect={selectTrade} expanded/>}
        <footer className="page-footer"><span><i className="dot cyan-bg"/>{data.source}</span><span>As of {datetime(data.asOf)}</span></footer>
      </>}
    </main>
    {settings && <div className="modal-backdrop" onClick={e => { if (e.target === e.currentTarget) setSettings(false); }}><section className="settings-modal" role="dialog" aria-modal="true" aria-labelledby="settings-title"><div className="panel-heading"><div><span className="eyebrow">YOUR DATA</span><h2 id="settings-title">Data & account settings</h2></div><button ref={closeSettings} className="icon-button" aria-label="Close data settings" onClick={() => setSettings(false)}><Icon name="close"/></button></div><p className="muted">Load a public wallet address. Your configured test address is read from private/address.txt on the server.</p><form onSubmit={event => { event.preventDefault(); portfolio.setWallet(walletInput.trim()); portfolio.setMode("live"); portfolio.refresh(); setSettings(false); }}><label className="field-label">Wallet address<input aria-label="Wallet address" value={walletInput} onChange={e => setWalletInput(e.target.value)} placeholder="Solana public address" required autoComplete="off" spellCheck={false}/></label><div className="settings-actions"><button className="primary-button" type="submit">Load live account <Icon name="arrow" size={14}/></button><button type="button" onClick={() => { portfolio.setMode("demo"); setSettings(false); }}>Use demo</button></div></form><div className="settings-divider"/><h3>Import historical performance</h3><p className="muted">Upload priced equity observations and external cash flows. Declare coverage and period-end flow timing; trades alone cannot establish portfolio returns.</p><button className="text-link" onClick={sampleHistory}>Download example JSON <Icon name="arrow" size={12}/></button><label className="import-zone">{importing ? "Validating history…" : "Choose portfolio history JSON"}<input aria-label="Import portfolio history" type="file" accept=".json,application/json" disabled={importing} onChange={async e => { const file = e.target.files?.[0]; if (!file) return; setImportError(""); setImporting(true); try { await portfolio.importHistory(file); } catch (error) { setImportError((error as Error).message); } finally { setImporting(false); e.target.value = ""; } }}/></label>{importError && <p className="negative" role="alert">{importError}</p>}<div className="settings-source"><h3>Source coverage</h3><p>Kamino: existing on-chain loader, one health calculation per obligation. Verify elevation-group terms and refreshed state against Kamino.</p><p>Activity: full wallet signature history, with a resumable background backfill; verified Jupiter, Orca and Raydium swaps, with actual token execution legs and native/wrapped SOL normalization. Composite actions remain unclassified.</p><p>Charts: Coinbase USD spot candles with UTC timestamps. Execution prices come from your transaction, rather than chart candles.</p><p>Live returns, realized P&L and historical loan health need a complete index of balances, cash flows, interest and cost basis.</p></div></section></div>}
  </div>;
}
