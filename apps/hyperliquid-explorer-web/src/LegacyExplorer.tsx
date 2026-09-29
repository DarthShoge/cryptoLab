import { useEffect, useState } from "react";
import {
  type Detail,
  type Run,
  type Scenario,
  runUrl,
  useResource,
} from "./api";
import { utc } from "./format";
import { Status } from "./Status";
import { Overview } from "./Overview";
import { Execution, Ledger } from "./Records";
import { DataView } from "./DataView";

function Workspace({ detail }: { detail: Detail }) {
  const [tab, setTab] = useState("Overview");
  const [scenario, setScenario] = useState<Scenario>(
    detail.scenarios.find(
      (s) => s.scenario_type === "strategy" && s.latency_seconds === 5,
    ) ?? detail.scenarios[0],
  );
  return (
    <>
      <div className="report-heading">
        <div>
          <p className="eyebrow">HYPERLIQUID / TRADER ENSEMBLE</p>
          <h1>Research explorer</h1>
          <p className="muted">
            {utc(detail.start)} → {utc(detail.end)}
          </p>
        </div>
        <span className={`mode-badge ${detail.synthetic ? "demo" : ""}`}>
          {detail.synthetic
            ? "SYNTHETIC DEMO"
            : detail.mode === "smoke_only"
              ? "SMOKE ONLY"
              : detail.mode === "historical_research"
                ? "RESEARCH REPORT"
                : "UNVERIFIED MODE"}
        </span>
      </div>
      <aside className="warnings" aria-label="Report warnings">
        <strong>Read this before interpreting performance</strong>
        {detail.synthetic && (
          <p>
            Fabricated wallets and prices. This demonstrates the pipeline, not
            investment performance.
          </p>
        )}
        {detail.warnings?.map((w, i) => (
          <p key={i}>{w}</p>
        ))}
      </aside>
      <nav className="tabs" role="tablist" aria-label="Report sections">
        {["Overview", "Traders", "Execution", "Data"].map((t) => (
          <button
            role="tab"
            aria-selected={tab === t}
            aria-controls="report-content"
            key={t}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </nav>
      <div id="report-content" role="tabpanel" aria-label={tab}>
        {tab === "Overview" && (
          <Overview
            detail={detail}
            selected={scenario}
            onChange={setScenario}
          />
        )}
        {tab === "Traders" && (
          <>
            <p className="muted">
              Historical wallet scores are not verified identities or a promise
              of future skill.
            </p>
            <Ledger id={detail.id} kind="traders" />
            <Ledger id={detail.id} kind="cohorts" />
          </>
        )}
        {tab === "Execution" && (
          <Execution
            detail={detail}
            scenario={scenario}
            onChange={setScenario}
          />
        )}
        {tab === "Data" && <DataView detail={detail} />}
      </div>
    </>
  );
}
export default function LegacyExplorer() {
  const [refresh, setRefresh] = useState(0),
    [id, setId] = useState("");
  const runs = useResource<Run[]>(`/api/runs?refresh=${refresh}`);
  useEffect(() => {
    if (runs.data && !runs.data.some((r) => r.id === id && r.available))
      setId(runs.data.find((r) => r.available)?.id ?? "");
  }, [runs.data, id]);
  const detail = useResource<Detail>(
    id ? runUrl(id, `?refresh=${refresh}`) : null,
  );
  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-icon">H</span>
          <span>
            CryptoLab <small>RESEARCH</small>
          </span>
        </div>
        <span className="read-only">
          <span /> Read-only · local artifacts
        </span>
      </header>
      <main>
        <p className="notice">
          <a href="/">← Copy-strategy lab</a> · Imported legacy artifacts.
          Missing configuration and universe evidence remain unavailable.
        </p>
        <div className="run-bar">
          <label>
            Saved run
            <select
              aria-label="Saved run"
              value={id}
              onChange={(e) => setId(e.target.value)}
            >
              <option value="" disabled>
                Select a report
              </option>
              {runs.data?.map((r) => (
                <option key={r.id} value={r.id} disabled={!r.available}>
                  {r.id.replace("hyperliquid_trader_ensemble_", "")}
                  {!r.available ? " · unavailable" : ""}
                </option>
              ))}
            </select>
          </label>
          <button onClick={() => setRefresh((n) => n + 1)}>
            ↻ Refresh reports
          </button>
        </div>
        <Status {...runs} />
        <Status {...detail} />
        {runs.data && !runs.data.some((r) => r.available) && (
          <div className="empty">
            <h1>Research explorer</h1>
            <h2>No available reports</h2>
            <p>
              Generate an offline report or point the Python API at its reports
              directory. No market data is downloaded by this app.
            </p>
          </div>
        )}
        {detail.data && (
          <Workspace key={`${id}:${refresh}`} detail={detail.data} />
        )}
      </main>
      <footer>
        CryptoLab / Hyperliquid ensemble{" "}
        <span>No live feed. No wallet connection. No exchange orders.</span>
      </footer>
    </div>
  );
}
