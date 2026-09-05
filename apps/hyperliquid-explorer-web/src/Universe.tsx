import { useState } from "react";
import { useResource } from "./api";
import { type Config, type UniversePage, type UniverseRow } from "./labApi";
import { formatValue, utc } from "./format";
import { Status } from "./Status";

const historyUrl = (id: string, params: URLSearchParams) =>
  `/api/lab/experiments/${id}/universe?${params}`;
function Breakdown({
  values,
}: {
  values?: Record<string, number | null> | null;
}) {
  return values ? (
    <details>
      <summary>View values</summary>
      <pre>{JSON.stringify(values, null, 2)}</pre>
    </details>
  ) : (
    <>—</>
  );
}

export function RankingTable({
  id,
  date = "",
  scope = "",
  wallet = "",
  onWallet,
}: {
  id: string;
  date?: string;
  scope?: string;
  wallet?: string;
  onWallet?: (wallet: string) => void;
}) {
  const [page, setPage] = useState(1),
    [selection, setSelection] = useState("");
  const params = new URLSearchParams({
    table: "rankings",
    page: String(page),
    page_size: "25",
  });
  for (const [key, value] of Object.entries({
    decision_date: date,
    scope,
    wallet,
    selection,
  }))
    if (value) params.set(key, value);
  const state = useResource<UniversePage>(historyUrl(id, params));
  return (
    <>
      <div className="toolbar compact">
        <label>
          Membership filter
          <select
            aria-label="Membership filter"
            value={selection}
            onChange={(e) => {
              setSelection(e.target.value);
              setPage(1);
            }}
          >
            <option value="">All observed candidates</option>
            <option value="selected">Selected</option>
            <option value="eligible">Eligible, below cutoff</option>
            <option value="excluded">Excluded by eligibility</option>
          </select>
        </label>
        <span className="muted">{state.data?.total ?? "—"} candidates</span>
      </div>
      <Status {...state} />
      {state.data?.rows.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Decision (UTC)</th>
                <th>Asset</th>
                <th>Rank</th>
                <th>Wallet</th>
                <th>Score</th>
                <th>Selected</th>
                <th>Weight</th>
                <th>Reasons</th>
                <th>Raw metrics</th>
                <th>Percentiles</th>
              </tr>
            </thead>
            <tbody>
              {state.data.rows.map((row, index) => (
                <tr key={index}>
                  <td className="nowrap">{utc(row.decision_time)}</td>
                  <td>{row.coin ?? "Pooled"}</td>
                  <td>{row.rank ?? "—"}</td>
                  <td>
                    {onWallet ? (
                      <button
                        className="wallet"
                        aria-label={`Inspect wallet ${row.user}`}
                        onClick={() => onWallet(row.user!)}
                        title={row.user ?? ""}
                      >
                        {row.user?.slice(0, 8)}…{row.user?.slice(-6)}
                      </button>
                    ) : (
                      <span className="wallet" title={row.user ?? ""}>
                        {row.user?.slice(0, 8)}…{row.user?.slice(-6)}
                      </span>
                    )}
                  </td>
                  <td>{formatValue(row.score, "ratio")}</td>
                  <td>
                    {row.selected
                      ? "Selected"
                      : row.eligible
                        ? "Below cutoff"
                        : "Excluded"}
                  </td>
                  <td>{formatValue(row.weight, "percent")}</td>
                  <td>{row.reasons?.join(", ") || "—"}</td>
                  <td>
                    <Breakdown values={row.metrics} />
                  </td>
                  <td>
                    <Breakdown values={row.percentiles} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        state.data && (
          <p className="empty">No matching historical candidates.</p>
        )
      )}
      <div className="pagination">
        <button disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
          Previous
        </button>
        <span>Page {page}</span>
        <button
          disabled={page * 25 >= (state.data?.total ?? 0)}
          onClick={() => setPage((p) => p + 1)}
        >
          Next
        </button>
      </div>
    </>
  );
}

function Contributions({
  id,
  config,
  wallet,
}: {
  id: string;
  config: Config;
  wallet: string;
}) {
  const [at, setAt] = useState(`${config.start}T00:00`),
    [coin, setCoin] = useState(config.coins[0]);
  const params = new URLSearchParams({
    table: "contributions",
    scope: coin,
    page_size: "200",
  });
  if (at) params.set("at", `${at}:00Z`);
  if (wallet) params.set("wallet", wallet);
  const state = useResource<UniversePage>(historyUrl(id, params));
  return (
    <section className="panel">
      <h2>From traders to portfolio target</h2>
      <p className="muted">
        Contributions explain target exposure, not allocated per-wallet trading
        PnL. Unknown positions retain their share of the coverage denominator.
      </p>
      <div className="toolbar">
        <label>
          Target timestamp (UTC)
          <input
            aria-label="Target timestamp (UTC)"
            type="datetime-local"
            value={at}
            onChange={(e) => setAt(e.target.value)}
          />
        </label>
        <label>
          Contribution asset
          <select
            aria-label="Contribution asset"
            value={coin}
            onChange={(e) => setCoin(e.target.value)}
          >
            {config.coins.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
        </label>
      </div>
      <Status {...state} />
      {state.data?.rows.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Wallet</th>
                <th>Native position</th>
                <th>Signal input</th>
                <th>Nominal weight</th>
                <th>Effective weight</th>
                <th>Target contribution</th>
                <th>Portfolio target</th>
                <th>Coverage</th>
              </tr>
            </thead>
            <tbody>
              {state.data.rows.map((r, i) => (
                <tr key={i}>
                  <td title={r.user ?? ""}>
                    {r.user?.slice(0, 8)}…{r.user?.slice(-6)}
                  </td>
                  <td>{r.position_qty ?? "Unknown"}</td>
                  <td>{formatValue(r.signal_input, "ratio")}</td>
                  <td>{formatValue(r.nominal_weight, "percent")}</td>
                  <td>{formatValue(r.effective_weight, "percent")}</td>
                  <td>{formatValue(r.target_contribution, "percent")}</td>
                  <td>{formatValue(r.portfolio_target, "percent")}</td>
                  <td>{r.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        state.data && (
          <p className="empty">
            No contribution rows at this timestamp/filter. Choose a configured
            target update; empty does not imply a known zero position.
          </p>
        )
      )}
    </section>
  );
}

export function Universe({ id, config }: { id: string; config: Config }) {
  const [date, setDate] = useState(config.start),
    [scope, setScope] = useState(
      config.scope === "pooled" ? "pooled" : config.coins[0],
    ),
    [wallet, setWallet] = useState(""),
    [cohortPage, setCohortPage] = useState(1);
  const cohorts = useResource<UniversePage>(
    historyUrl(
      id,
      new URLSearchParams({
        table: "cohorts",
        page_size: "50",
        page: String(cohortPage),
      }),
    ),
  );
  return (
    <>
      <section className="panel">
        <h2>Historical trader universe</h2>
        <p className="muted">
          Each row is an actual selection decision using strictly prior
          information. Membership turnover is separate from portfolio trading
          turnover.
        </p>
        <Status {...cohorts} />
        {cohorts.data && (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Decision (UTC)</th>
                  <th>Scope</th>
                  <th>Observed</th>
                  <th>Eligible</th>
                  <th>Requested</th>
                  <th>Selected</th>
                  <th>Entries / exits</th>
                  <th>Retention</th>
                  <th>Membership turnover</th>
                </tr>
              </thead>
              <tbody>
                {cohorts.data.rows.map((r, i) => (
                  <tr
                    key={i}
                    className={
                      r.decision_time?.slice(0, 10) === date &&
                      (r.coin ?? "pooled") === scope
                        ? "selected-row"
                        : ""
                    }
                  >
                    <td>
                      <button
                        className="link-button"
                        onClick={() => {
                          setDate(r.decision_time!.slice(0, 10));
                          setScope(r.coin ?? "pooled");
                        }}
                      >
                        {utc(r.decision_time)}
                      </button>
                    </td>
                    <td>{r.coin ?? "Pooled"}</td>
                    <td>{r.candidate_count}</td>
                    <td>{r.eligible_count}</td>
                    <td>{r.requested_count}</td>
                    <td>{r.selected_count}</td>
                    <td
                      title={`Entries: ${r.entries?.join(", ")}\nExits: ${r.exits?.join(", ")}`}
                    >
                      +{r.entries?.length} / −{r.exits?.length}
                    </td>
                    <td>{formatValue(r.retention, "percent")}</td>
                    <td>{formatValue(r.membership_turnover, "percent")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="pagination">
          <button
            disabled={cohortPage === 1}
            onClick={() => setCohortPage((p) => p - 1)}
          >
            Earlier page
          </button>
          <span>{cohorts.data?.total ?? 0} cohort decisions</span>
          <button
            disabled={cohortPage * 50 >= (cohorts.data?.total ?? 0)}
            onClick={() => setCohortPage((p) => p + 1)}
          >
            More decisions
          </button>
        </div>
      </section>
      <section className="panel">
        <h2>Selection decision drilldown</h2>
        <div className="toolbar">
          <label>
            Decision date
            <input
              aria-label="Decision date"
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
            />
          </label>
          <label>
            Universe scope
            <select
              aria-label="Universe scope"
              value={scope}
              onChange={(e) => setScope(e.target.value)}
            >
              {(config.scope === "pooled" ? ["pooled"] : config.coins).map(
                (c) => (
                  <option key={c}>{c}</option>
                ),
              )}
            </select>
          </label>
        </div>
        <RankingTable
          key={`${date}:${scope}`}
          id={id}
          date={date}
          scope={scope}
          onWallet={setWallet}
        />
      </section>
      {wallet && (
        <section className="panel">
          <div className="section-heading">
            <h2>Wallet membership history</h2>
            <button onClick={() => setWallet("")}>Clear wallet</button>
          </div>
          <p className="wallet-address">{wallet}</p>
          <RankingTable key={wallet} id={id} wallet={wallet} />
        </section>
      )}
      <Contributions key={wallet} id={id} config={config} wallet={wallet} />
    </>
  );
}
