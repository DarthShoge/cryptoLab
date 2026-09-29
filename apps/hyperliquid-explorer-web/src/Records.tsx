import { useState } from "react";
import {
  type Detail,
  type Row,
  type Scenario,
  type RecordPage,
  runUrl,
  scenarioParams,
  useResource,
} from "./api";
import { formatValue, label, utc } from "./format";
import { ScenarioSelect } from "./Overview";
import { Status } from "./Status";

function Cell({ value, field }: { value: unknown; field: string }) {
  const [copied, setCopied] = useState(false);
  if (value == null) return <>—</>;
  if (field === "user" && typeof value === "string")
    return (
      <button
        className="wallet"
        title={value}
        aria-label={`Copy wallet ${value}`}
        onClick={() =>
          navigator.clipboard
            .writeText(value)
            .then(() => setCopied(true))
            .catch(() => setCopied(false))
        }
      >
        {value.slice(0, 8)}…{value.slice(-6)} {copied ? "✓" : "⧉"}
      </button>
    );
  if (field.endsWith("time"))
    return <span className="nowrap">{utc(String(value))}</span>;
  if (typeof value === "boolean")
    return (
      <span className={value ? "pill positive" : "muted"}>
        {value ? "Selected" : "Excluded"}
      </span>
    );
  if (Array.isArray(value))
    return (
      <span title={value.join(", ")}>
        {value.length
          ? value
              .map(String)
              .map((s) =>
                s.startsWith("0x") ? `${s.slice(0, 6)}…${s.slice(-4)}` : s,
              )
              .join(", ")
          : "—"}
      </span>
    );
  if (typeof value === "number")
    return <>{value.toLocaleString("en-US", { maximumFractionDigits: 6 })}</>;
  if (typeof value === "object")
    return (
      <details>
        <summary>Metric breakdown</summary>
        <pre>{JSON.stringify(value, null, 2)}</pre>
      </details>
    );
  return <>{String(value)}</>;
}
export function Ledger({
  id,
  kind,
  scenario,
}: {
  id: string;
  kind: "traders" | "cohorts" | "fills" | "funding";
  scenario?: Scenario;
}) {
  const [page, setPage] = useState(1),
    [wallet, setWallet] = useState(""),
    [scope, setScope] = useState(""),
    [date, setDate] = useState(""),
    [coin, setCoin] = useState(""),
    [reason, setReason] = useState("");
  const params = scenario ? scenarioParams(scenario) : new URLSearchParams();
  params.set("page", String(page));
  params.set("page_size", "25");
  for (const [k, v] of Object.entries({
    wallet,
    scope,
    decision_date: date,
    coin,
    reason,
  }))
    if (v) params.set(k, v);
  const state = useResource<RecordPage>(runUrl(id, `/${kind}?${params}`));
  const columns: Record<typeof kind, (keyof Row)[]> = {
    traders: [
      "decision_time",
      "user",
      "coin",
      "score",
      "selected",
      "exclusions",
      "metrics",
    ],
    cohorts: ["decision_time", "coin", "members", "cutoff_address"],
    fills: [
      "signal_time",
      "coin",
      "requested_qty",
      "filled_qty",
      "vwap",
      "book_time",
      "fee",
      "reason",
    ],
    funding: ["time", "coin", "qty", "mark", "rate", "cash_delta"],
  };
  const titles = {
    traders: "Trader rankings",
    cohorts: "Cohort history",
    fills: "Simulated fills",
    funding: "Funding ledger",
  };
  return (
    <section className="panel">
      <div className="section-heading">
        <h2>{titles[kind]}</h2>
        <span className="pill">{state.data?.total ?? "—"} rows</span>
      </div>
      <div className="toolbar compact">
        {kind === "traders" && (
          <label>
            Wallet search
            <input
              value={wallet}
              maxLength={42}
              placeholder="0x…"
              onChange={(e) => {
                setWallet(e.target.value);
                setPage(1);
              }}
            />
          </label>
        )}
        {kind === "traders" || kind === "cohorts" ? (
          <>
            <label>
              Decision date (UTC)
              <input
                type="date"
                value={date}
                onChange={(e) => {
                  setDate(e.target.value);
                  setPage(1);
                }}
              />
            </label>
            <label>
              Score scope
              <select
                value={scope}
                onChange={(e) => {
                  setScope(e.target.value);
                  setPage(1);
                }}
              >
                <option value="">All scopes</option>
                <option value="global">Global</option>
                {["BTC", "ETH", "SOL"].map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
            </label>
          </>
        ) : (
          <label>
            Asset
            <select
              value={coin}
              onChange={(e) => {
                setCoin(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All assets</option>
              {["BTC", "ETH", "SOL"].map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
          </label>
        )}
        {kind === "fills" && (
          <label>
            Execution outcome
            <select
              value={reason}
              onChange={(e) => {
                setReason(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All outcomes</option>
              {["filled", "partial_depth", "stale_book"].map((r) => (
                <option key={r}>{r}</option>
              ))}
            </select>
          </label>
        )}
      </div>
      <Status {...state} />
      {state.data &&
        (!state.data.available ? (
          <p className="empty">{state.data.reason}</p>
        ) : state.data.rows.length === 0 ? (
          <p className="empty">
            No matching rows. Empty does not imply missing data or zero costs.
          </p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  {columns[kind].map((c) => (
                    <th key={c}>{label(c)}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {state.data.rows.map((r, i) => (
                  <tr key={i}>
                    {columns[kind].map((c) => (
                      <td key={c}>
                        {c === "coin" &&
                        r[c] === null &&
                        (kind === "traders" || kind === "cohorts") ? (
                          "Global"
                        ) : (
                          <Cell value={r[c]} field={c} />
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      <div className="pagination">
        <button
          disabled={page === 1 || state.loading}
          onClick={() => setPage((p) => p - 1)}
        >
          Previous
        </button>
        <span>
          Page {page} of {Math.max(1, Math.ceil((state.data?.total ?? 0) / 25))}
        </span>
        <button
          disabled={state.loading || page * 25 >= (state.data?.total ?? 0)}
          onClick={() => setPage((p) => p + 1)}
        >
          Next
        </button>
      </div>
    </section>
  );
}
export function Execution({
  detail,
  scenario,
  onChange,
  strategyLabel,
}: {
  detail: Detail;
  scenario: Scenario;
  onChange: (s: Scenario) => void;
  strategyLabel?: string;
}) {
  return (
    <>
      <div className="toolbar">
        <ScenarioSelect
          scenarios={detail.scenarios}
          selected={scenario}
          onChange={onChange}
          strategyLabel={strategyLabel}
        />
      </div>
      <p className="notice">
        Simulated execution only. Funding uses a historical mid-price
        approximation. No exchange orders are submitted.
      </p>
      <Ledger
        key={`fills:${scenario.name}:${scenario.latency_seconds}`}
        id={detail.id}
        kind="fills"
        scenario={scenario}
      />
      <Ledger
        key={`funding:${scenario.name}:${scenario.latency_seconds}`}
        id={detail.id}
        kind="funding"
        scenario={scenario}
      />
      <section className="panel">
        <h2>Residual positions</h2>
        <p className="muted">
          Native quantity and average entry; open positions are not
          force-closed.
        </p>
        {scenario.positions?.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Asset</th>
                  <th>Native quantity</th>
                  <th>Average entry</th>
                </tr>
              </thead>
              <tbody>
                {scenario.positions.map((p) => (
                  <tr key={p.coin}>
                    <td>{p.coin}</td>
                    <td>{p.qty}</td>
                    <td>{formatValue(p.entry, "usd")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="empty">No residual positions recorded.</p>
        )}
      </section>
    </>
  );
}
