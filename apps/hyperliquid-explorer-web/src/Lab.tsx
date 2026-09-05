import { useState } from "react";
import { useResource } from "./api";
import { type Bootstrap, type Dataset, type Submission } from "./labApi";
import { Status } from "./Status";
import { StrategyBuilder } from "./StrategyBuilder";
import { ExperimentDetail, Experiments } from "./Experiments";
import { Compare } from "./Compare";

export function Lab() {
  const bootstrap = useResource<Bootstrap>("/api/lab/bootstrap");
  const datasets = useResource<Dataset[]>("/api/lab/datasets");
  const [view, setView] = useState("builder"),
    [id, setId] = useState<string | null>(null),
    [comparison, setComparison] = useState<string[]>([]);
  const [draft, setDraft] = useState<Submission | undefined>(),
    [draftKey, setDraftKey] = useState(0);
  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="/">
          <span className="brand-icon">H</span>
          <span>
            CryptoLab <small>HYPERLIQUID</small>
          </span>
        </a>
        <span className="read-only">
          <span /> Local simulations · no exchange orders
        </span>
      </header>
      <main>
        <div className="report-heading">
          <div>
            <p className="eyebrow">COPY THE TRADERS · TEST THE RULE</p>
            <h1>Hyperliquid copy-strategy lab</h1>
            <p className="muted">
              Define your trader universe. Backtest the copying rule. Save and
              compare the evidence.
            </p>
          </div>
          <a className="legacy-link" href="/reports">
            Imported reports ↗
          </a>
        </div>
        <nav className="lab-nav" aria-label="Strategy lab">
          <button
            className={view === "builder" ? "active" : ""}
            onClick={() => setView("builder")}
          >
            Strategy builder
          </button>
          <button
            className={view === "library" ? "active" : ""}
            onClick={() => setView("library")}
          >
            Saved backtests
          </button>
          {id && (
            <button
              className={view === "detail" ? "active" : ""}
              onClick={() => setView("detail")}
            >
              Current backtest
            </button>
          )}
          {comparison.length > 1 && (
            <button
              className={view === "compare" ? "active" : ""}
              onClick={() => setView("compare")}
            >
              Comparison
            </button>
          )}
        </nav>
        <Status {...bootstrap} />
        <Status {...datasets} />
        {bootstrap.data && datasets.data && (
          <>
            <div hidden={view !== "builder"}>
              <StrategyBuilder
                key={draftKey}
                bootstrap={bootstrap.data}
                datasets={datasets.data}
                initial={draft}
                onRun={(e) => {
                  setId(e.id);
                  setView("detail");
                }}
              />
            </div>
            {view === "library" && (
              <Experiments
                onOpen={(value) => {
                  setId(value);
                  setView("detail");
                }}
                onCompare={(values) => {
                  setComparison(values);
                  setView("compare");
                }}
              />
            )}
            {view === "detail" && id && (
              <ExperimentDetail
                key={id}
                id={id}
                token={bootstrap.data.token}
                onClone={(value) => {
                  setDraft(value);
                  setDraftKey((n) => n + 1);
                  setView("builder");
                }}
              />
            )}
            {view === "compare" && <Compare ids={comparison} />}
          </>
        )}
        <details className="capabilities">
          <summary>Research readiness & boundaries</summary>
          {bootstrap.data?.restrictions.map((r) => (
            <p key={r}>{r}</p>
          ))}
        </details>
      </main>
      <footer>
        Hyperliquid copy-strategy research
        <span>
          Immutable backtests · historical selection evidence · no live trading
        </span>
      </footer>
    </div>
  );
}
