import { useEffect, useState } from "react";
import { type Detail, type Scenario, runUrl, useResource } from "./api";
import {
  type Config,
  type Experiment,
  type Submission,
  mutate,
  strategySummary,
  useExperiment,
} from "./labApi";
import { Status } from "./Status";
import { Overview } from "./Overview";
import { Execution } from "./Records";
import { DataView } from "./DataView";
import { Universe } from "./Universe";
import { utc } from "./format";

export function Experiments({
  onOpen,
  onCompare,
}: {
  onOpen: (id: string) => void;
  onCompare: (ids: string[]) => void;
}) {
  const [refresh, setRefresh] = useState(0),
    [selected, setSelected] = useState<string[]>([]);
  const state = useResource<Experiment[]>(
    `/api/lab/experiments?refresh=${refresh}`,
  );
  useEffect(() => {
    if (
      state.data?.some((e) => e.status === "queued" || e.status === "running")
    ) {
      const timer = setTimeout(() => setRefresh((n) => n + 1), 1500);
      return () => clearTimeout(timer);
    }
  }, [state.data]);
  return (
    <section className="panel">
      <div className="section-heading">
        <div>
          <h2>Saved backtests</h2>
          <p className="muted">
            Every submitted hypothesis is saved automatically. Clone to vary
            parameters; results are never overwritten.
          </p>
        </div>
        <button onClick={() => setRefresh((n) => n + 1)}>
          Refresh library
        </button>
      </div>
      <Status {...state} />
      {state.data?.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Compare</th>
                <th>Hypothesis</th>
                <th>Strategy definition</th>
                <th>Period</th>
                <th>Status</th>
                <th>Saved</th>
              </tr>
            </thead>
            <tbody>
              {state.data.map((e) => (
                <tr key={e.id}>
                  <td>
                    <input
                      aria-label={`Compare ${e.name}`}
                      type="checkbox"
                      disabled={
                        e.status !== "completed" ||
                        (!selected.includes(e.id) && selected.length >= 6)
                      }
                      checked={selected.includes(e.id)}
                      onChange={(event) =>
                        setSelected((ids) =>
                          event.target.checked
                            ? [...ids, e.id]
                            : ids.filter((id) => id !== e.id),
                        )
                      }
                    />
                  </td>
                  <td>
                    <button
                      className="link-button"
                      onClick={() => onOpen(e.id)}
                    >
                      {e.name}
                    </button>
                    <small>
                      {e.provenance.manifest &&
                      typeof e.provenance.manifest === "object" &&
                      !Array.isArray(e.provenance.manifest) &&
                      "synthetic" in e.provenance.manifest &&
                      e.provenance.manifest.synthetic === true
                        ? "SYNTHETIC"
                        : "DEVELOPMENT ONLY"}
                    </small>
                  </td>
                  <td className="strategy-cell">
                    {strategySummary(e.config as Config)}
                  </td>
                  <td className="nowrap">
                    {e.config.start} → {e.config.end}
                  </td>
                  <td>
                    {e.status}
                    {e.needs_resume ? " · resume required" : ""}
                  </td>
                  <td className="nowrap">{utc(e.created_at)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        state.data && (
          <p className="empty">
            No saved backtests yet. Configure a hypothesis and run it to start
            your experiment history.
          </p>
        )
      )}
      <div className="toolbar">
        <button
          className="primary"
          disabled={selected.length < 2}
          onClick={() => onCompare(selected)}
        >
          Compare selected
        </button>
        <span className="muted">{selected.length} / 6 selected</span>
      </div>
    </section>
  );
}

function ResultPanels({ experiment }: { experiment: Experiment }) {
  const [tab, setTab] = useState("Performance");
  const report = useResource<Detail>(
    experiment.run_id ? runUrl(experiment.run_id) : null,
  );
  const [selected, setSelected] = useState<Scenario | null>(null);
  const scenario =
    selected ??
    report.data?.scenarios.find((s) => s.scenario_type === "strategy");
  return (
    <>
      <nav className="tabs" role="tablist" aria-label="Backtest research views">
        {[
          "Performance",
          "Trader universe",
          "Execution",
          "Data & assumptions",
        ].map((t) => (
          <button
            role="tab"
            aria-selected={tab === t}
            key={t}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </nav>
      <Status {...report} />
      {tab === "Trader universe" && (
        <Universe id={experiment.id} config={experiment.config as Config} />
      )}
      {tab === "Performance" && report.data && scenario && (
        <Overview
          detail={report.data}
          selected={scenario}
          onChange={setSelected}
        />
      )}
      {tab === "Execution" && report.data && scenario && (
        <Execution
          detail={report.data}
          scenario={scenario}
          onChange={setSelected}
        />
      )}
      {tab === "Data & assumptions" && report.data && (
        <>
          <DataView detail={report.data} />
          <section className="panel">
            <h2>Immutable experiment evidence</h2>
            <pre>
              {JSON.stringify(
                {
                  config_hash: experiment.config_hash,
                  provenance: experiment.provenance,
                  artifact_hashes: experiment.artifact_hashes,
                },
                null,
                2,
              )}
            </pre>
          </section>
        </>
      )}
    </>
  );
}

export function ExperimentDetail({
  id,
  token,
  onClone,
}: {
  id: string;
  token: string;
  onClone: (s: Submission) => void;
}) {
  const state = useExperiment(id);
  const [error, setError] = useState(""),
    [editing, setEditing] = useState(false),
    [name, setName] = useState(""),
    [notes, setNotes] = useState("");
  const e = state.data;
  const act = async (action: string) => {
    setError("");
    try {
      if (action === "clone")
        onClone(await mutate<Submission>(`/experiments/${id}/clone`, token));
      else {
        await mutate(`/experiments/${id}/${action}`, token);
        state.retry();
      }
    } catch (ex) {
      setError(String(ex));
    }
  };
  const save = async () => {
    try {
      await mutate(
        `/experiments/${id}/metadata`,
        token,
        { name, notes },
        "PATCH",
      );
      setEditing(false);
      state.retry();
    } catch (ex) {
      setError(String(ex));
    }
  };
  return (
    <>
      <Status {...state} />
      {e && (
        <>
          <div className="report-heading">
            <div>
              <p className="eyebrow">SAVED COPY-STRATEGY EXPERIMENT</p>
              <h2 className="experiment-title">{e.name}</h2>
              <p className="muted">
                {e.config.start} → {e.config.end} · {e.id.slice(0, 10)} ·
                development only
              </p>
            </div>
            <span className="mode-badge">{e.status}</span>
          </div>
          <div className="hypothesis-strip">
            <p>{strategySummary(e.config as Config)}</p>
            <small>Frozen configuration · {e.config_hash.slice(0, 16)}</small>
          </div>
          <div className="toolbar">
            <button onClick={() => void act("clone")}>
              Clone configuration
            </button>
            <button
              onClick={() => {
                setName(e.name);
                setNotes(e.notes);
                setEditing(true);
              }}
            >
              Edit name / notes
            </button>
            {["queued", "running"].includes(e.status) && (
              <button onClick={() => void act("cancel")}>Cancel job</button>
            )}
            {e.needs_resume && e.status === "queued" && (
              <button onClick={() => void act("resume")}>
                Resume queued job
              </button>
            )}
          </div>
          {editing && (
            <section className="panel">
              <label>
                Saved name
                <input
                  aria-label="Saved name"
                  value={name}
                  maxLength={120}
                  onChange={(event) => setName(event.target.value)}
                />
              </label>
              <label>
                Research notes
                <textarea
                  aria-label="Research notes"
                  value={notes}
                  maxLength={4000}
                  onChange={(event) => setNotes(event.target.value)}
                />
              </label>
              <button onClick={() => void save()}>Save annotations</button>
            </section>
          )}
          {e.notes && <p className="notice">{e.notes}</p>}
          {(error || e.error) && (
            <p role="alert" className="notice error">
              {error || e.error}
            </p>
          )}
          {e.status === "queued" || e.status === "running" ? (
            <p className="notice" role="status">
              {e.needs_resume
                ? "Saved queued job awaits explicit resume."
                : "Running a bounded local simulation. Your configuration is already saved."}
            </p>
          ) : (
            e.status === "completed" && <ResultPanels experiment={e} />
          )}
        </>
      )}
    </>
  );
}
