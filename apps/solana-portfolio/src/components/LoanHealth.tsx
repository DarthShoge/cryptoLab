import { useState } from "react";
import type { Loan, Portfolio } from "../types";
import { money, number, percent, short } from "../format";
import { Icon } from "./Icon";

export function LoanHealth({ data }: { data: Portfolio }) {
  const [selected, setSelected] = useState(0);
  const loans = data.summary.loans;
  const loan: Loan | undefined = loans[Math.min(selected, loans.length - 1)];
  const utilization = loan?.ltv != null && loan.liquidationLtv != null && loan.liquidationLtv > 0 ? loan.ltv / loan.liquidationLtv : 0;
  const state = loan?.health == null ? loan && loan.debt === 0 ? "NO DEBT" : "UNAVAILABLE" : loan.health < 1 ? "LIQUIDATABLE" : loan.health < 1.15 ? "WATCH CLOSELY" : "ABOVE THRESHOLD";
  return <section className="panel health-panel"><div className="panel-heading"><h2>Kamino health</h2><span className="protocol-mark">K</span></div>
    {!loan ? <div className="empty-health"><Icon name="positions" size={28}/><p>No Kamino obligation loaded</p><span className="muted">Load your wallet to inspect collateral and debt.</span></div> : <>
      <div className="health-top"><span className="muted">Liquidation health factor</span><span className={`badge ${utilization >= 1 ? "danger-badge" : utilization >= .87 ? "warning-badge" : "good-badge"}`}>{state}</span></div>
      <div className={`health-number ${utilization >= .87 ? "orange" : "lime"}`}>{loan.health === null && loan.debt === 0 ? "∞" : number(loan.health, 3)}<span> / 1.00 threshold</span></div>
      <div className="health-track"><div style={{ width: `${Math.min(100, utilization * 100)}%` }} className={utilization >= .87 ? "orange-bg" : "cyan-bg"}/><i style={{ left: "100%" }}/></div><div className="track-labels"><span>0% LTV</span><span>Liquidation {percent(loan.liquidationLtv == null ? null : loan.liquidationLtv * 100)}</span></div>
      <div className="health-details"><div><span>Risk-adjusted LTV</span><strong className="cyan">{percent(loan.ltv == null ? null : loan.ltv * 100)}</strong></div><div><span>Liquidation buffer</span><strong className={loan.liquidationBuffer !== null && loan.liquidationBuffer < 0 ? "negative" : ""}>{money(loan.liquidationBuffer)}</strong></div><div><span>Collateral supplied</span><strong>{money(loan.supplied)}</strong></div><div><span>Total borrowed</span><strong className="orange">{money(loan.debt)}</strong></div><div><span>Borrow-capacity factor</span><strong>{number(loan.borrowHealth, 3)}</strong></div></div>
      <div className="health-notice"><i className="dot orange-bg"/><p>Health belongs to this obligation. Assets in other markets or your wallet do not protect it.</p></div>
      {loans.length > 1 && <label className="loan-select">Obligation<select aria-label="Kamino obligation" value={Math.min(selected, loans.length - 1)} onChange={e => setSelected(Number(e.target.value))}>{loans.map((item, i) => <option key={item.address} value={i}>{short(item.address, 6)}</option>)}</select></label>}
      <div className="health-bottom"><span className="muted">{short(loan.address, 6)}</span>{data.mode === "live" && <a href={`https://app.kamino.finance/`} target="_blank" rel="noreferrer">Open Kamino <Icon name="external" size={12}/></a>}</div>
    </>}
  </section>;
}
