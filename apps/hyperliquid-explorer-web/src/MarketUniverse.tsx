import { useState } from "react";
import type { components } from "./api.generated";
import { useResource } from "./api";
import { formatValue, utc } from "./format";
import { Status } from "./Status";
import { classes } from "./marketConfig";

type MarketPage = components["schemas"]["Page_MarketRow_"];
export function MarketUniverse({
  id,
  start,
  onTraders,
}: {
  id: string;
  start: string;
  onTraders?: (date: string, instrument: string) => void;
}) {
  const [date, setDate] = useState(start),
    [assetClass, setAssetClass] = useState(""),
    [instrument, setInstrument] = useState(""),
    [page, setPage] = useState(1),
    [cohortPage, setCohortPage] = useState(1);
  const base = `/api/lab/experiments/${id}/market-universe?`;
  const cohorts = useResource<MarketPage>(
    base +
      new URLSearchParams({
        table: "cohorts",
        page: String(cohortPage),
        page_size: "25",
      }),
  );
  const params = new URLSearchParams({
    table: "rankings",
    page: String(page),
    page_size: "25",
  });
  if (date) params.set("decision_date", date);
  if (assetClass) params.set("asset_class", assetClass);
  if (instrument) params.set("instrument_id", instrument);
  const rankings = useResource<MarketPage>(base + params);
  return (
    <>
      <section className="panel">
        <h2>Historical market universe</h2>
        <p className="muted">
          Actual market decisions, before trader ranking. Membership turnover is
          not portfolio trading turnover. Instrument exits continue normal
          liquidation and residual-position accounting.
        </p>
        <Status {...cohorts} />
        {cohorts.data?.available === false && (
          <p className="notice">{cohorts.data.reason}</p>
        )}
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Decision</th>
                <th>Selected / requested</th>
                <th>Members</th>
                <th>Entries</th>
                <th>Exits</th>
                <th>Market turnover</th>
              </tr>
            </thead>
            <tbody>
              {cohorts.data?.rows.map((row) => (
                <tr key={row.decision_time}>
                  <td>
                    <button
                      className="link-button"
                      onClick={() => {
                        setDate(row.decision_time.slice(0, 10));
                        setPage(1);
                      }}
                    >
                      {utc(row.decision_time)}
                    </button>
                  </td>
                  <td>
                    {row.selected_count} / {row.requested_count}
                  </td>
                  <td>{row.members?.join(", ") || "Cash"}</td>
                  <td>{row.entries?.join(", ") || "—"}</td>
                  <td>{row.exits?.join(", ") || "—"}</td>
                  <td>{formatValue(row.membership_turnover, "percent")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="pagination">
          <button
            disabled={cohortPage === 1}
            onClick={() => setCohortPage((p) => p - 1)}
          >
            Previous decisions
          </button>
          <span>{cohorts.data?.total ?? 0} market decisions</span>
          <button
            disabled={cohortPage * 25 >= (cohorts.data?.total ?? 0)}
            onClick={() => setCohortPage((p) => p + 1)}
          >
            More market decisions
          </button>
        </div>
      </section>
      <section className="panel">
        <h2>Market selection drilldown</h2>
        <div className="toolbar">
          <label>
            Market decision date
            <input
              aria-label="Market decision date"
              type="date"
              value={date}
              onChange={(e) => {
                setDate(e.target.value);
                setPage(1);
              }}
            />
          </label>
          <label>
            Market class filter
            <select
              aria-label="Market class filter"
              value={assetClass}
              onChange={(e) => {
                setAssetClass(e.target.value);
                setPage(1);
              }}
            >
              <option value="">All classes</option>
              {classes.map(([id, name]) => (
                <option value={id} key={id}>
                  {name}
                </option>
              ))}
            </select>
          </label>
          <label>
            Exact instrument ID
            <input
              aria-label="Exact instrument ID"
              value={instrument}
              onChange={(e) => {
                setInstrument(e.target.value);
                setPage(1);
              }}
            />
          </label>
        </div>
        <Status {...rankings} />
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Instrument</th>
                <th>Class / venue</th>
                <th>Rank</th>
                <th>USD volume</th>
                <th>Prior window</th>
                <th>Budget</th>
                <th>Selection / reasons</th>
                <th>Trader evidence</th>
              </tr>
            </thead>
            <tbody>
              {rankings.data?.rows.map((row) => (
                <tr key={`${row.decision_time}:${row.instrument_id}`}>
                  <td>
                    {row.instrument_id}
                    <small>{row.display_name}</small>
                    {row.proxy_ticker && <small>Proxy: {row.proxy_ticker}</small>}
                    {row.availability_basis && <small>{row.availability_basis.replace(/_/g, " ")}</small>}
                  </td>
                  <td>
                    {row.asset_class} / {row.venue}
                  </td>
                  <td>{row.rank ?? "—"}</td>
                  <td>{formatValue(row.volume_usd, "usd")}</td>
                  <td>
                    {row.window_start?.slice(0, 10) ?? "—"} →{" "}
                    {row.window_end?.slice(0, 10) ?? "—"}
                  </td>
                  <td>{formatValue(row.budget, "percent")}</td>
                  <td>
                    {row.selected
                      ? "Selected"
                      : row.reasons?.join(", ") || "Below cutoff"}
                  </td>
                  <td>
                    {row.selected && onTraders && (
                      <button
                        onClick={() =>
                          onTraders(
                            row.decision_time.slice(0, 10),
                            row.instrument_id!,
                          )
                        }
                      >
                        Inspect traders
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {rankings.data?.total === 0 && <p>No matching market decisions.</p>}
        <div className="pagination">
          <button disabled={page === 1} onClick={() => setPage((p) => p - 1)}>
            Previous markets
          </button>
          <span>{rankings.data?.total ?? 0} candidate rows</span>
          <button
            disabled={page * 25 >= (rankings.data?.total ?? 0)}
            onClick={() => setPage((p) => p + 1)}
          >
            More markets
          </button>
        </div>
      </section>
    </>
  );
}
