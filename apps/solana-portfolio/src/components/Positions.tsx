import { useState } from "react";
import type { Portfolio } from "../types";
import { money, number, percent, short } from "../format";
export function Positions({ data, expanded = false, onViewAll }: { data: Portfolio; expanded?: boolean; onViewAll?: () => void }) {
  const [tab, setTab] = useState<"all" | "supplied" | "borrowed" | "wallet">("all");
  const [search, setSearch] = useState("");
  const positions = data.summary.positions.filter(p => (tab === "all" || p.kind === tab) && `${p.symbol} ${p.mint || ""} ${p.obligation || ""}`.toLowerCase().includes(search.toLowerCase()));
  const rank = { supplied: 0, borrowed: 1, wallet: 2 };
  positions.sort((a, b) => rank[a.kind] - rank[b.kind] || (b.value ?? -1) - (a.value ?? -1));
  const visible = expanded ? positions : positions.slice(0, 8);
  const assets = Object.entries(data.summary.exposure);
  return <section className="panel positions-panel"><div className="panel-heading"><h2>Positions <span className="muted mini">{data.summary.positions.length} POSITIONS</span></h2><span className="text-link">Collateral + debt + wallet</span></div>
    <div className="positions-summary"><div><span>Total supplied</span><strong>{money(data.summary.supplied)}</strong></div><div><span>Total borrowed</span><strong className="orange">{money(data.summary.debt)}</strong></div><div><span>Wallet assets</span><strong>{money(data.summary.walletValue)}</strong></div></div>
    <div className="segmented position-tabs">{(["all", "supplied", "borrowed", "wallet"] as const).map(value => <button key={value} className={tab === value ? "active" : ""} onClick={() => setTab(value)}>{value === "all" ? "All positions" : value}</button>)}</div>
    {expanded && <div className="position-search"><input aria-label="Search positions" placeholder="Search asset, mint or obligation…" value={search} onChange={e => setSearch(e.target.value)}/></div>}
    <div className="table-scroll"><table><thead><tr><th>ASSET / LOCATION</th><th>AMOUNT</th><th>VALUE</th><th>APY</th></tr></thead><tbody>{visible.map(position => <tr key={position.id}><td><div className="asset-cell"><span className={`token token-${position.symbol.toLowerCase()}`}>{position.symbol === "SOL" ? "≋" : position.symbol === "ETH" ? "◆" : position.symbol === "BTC" ? "₿" : ["USDC", "USDT"].includes(position.symbol) ? "$" : "?"}</span><div><strong>{position.symbol}</strong><small>{position.kind === "wallet" ? "Wallet" : `Kamino · ${position.kind}`} {expanded && position.obligation ? `· ${short(position.obligation)}` : ""}</small></div></div></td><td className="mono">{number(position.amount, 4)}</td><td className={`mono ${position.kind === "borrowed" ? "orange" : ""}`}>{position.kind === "borrowed" ? "−" : ""}{money(position.value)}</td><td className={position.kind === "borrowed" ? "orange mono" : "lime mono"}>{percent(position.apy)}</td></tr>)}</tbody></table></div>
    {!positions.length && <p className="empty-state">No positions available in this view.</p>}
    {!expanded && positions.length > visible.length && <div className="positions-more"><button className="text-link" onClick={onViewAll}>View all {data.summary.positions.length} positions</button><span className="muted">{positions.length - visible.length} more in this filter</span></div>}
    <div className="exposure-strip"><span className="muted">NET TOKEN EXPOSURE</span>{assets.slice(0, expanded ? assets.length : 4).map(([asset, amount]) => <span key={asset}><i className={`dot ${amount < 0 ? "orange-bg" : "cyan-bg"}`}/><strong className={amount < 0 ? "orange" : "cyan"}>{number(amount, 3)}</strong> {asset}</span>)}</div>
  </section>;
}
