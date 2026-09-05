import { useState } from "react";
import { useResource } from "./api";
import type { Comparison } from "./labApi";
import { formatValue, label } from "./format";
import { Status } from "./Status";
import { Plot } from "./Charts";

const colors = [
  "#7fe2ca",
  "#8b9de8",
  "#f2ab89",
  "#d5b1ec",
  "#d6d97c",
  "#82c7e8",
];
export function Compare({ ids }: { ids: string[] }) {
  const [units, setUnits] = useState("growth");
  const state = useResource<Comparison>(
    `/api/lab/compare?${new URLSearchParams({ ids: ids.join(","), units })}`,
  );
  const data = state.data;
  const differences =
    data?.differences.flatMap((key) => {
      if (key !== "trader" && key !== "follower")
        return [
          {
            key,
            title: label(key),
            values: data.series.map((s) => s.config[key]),
          },
        ];
      const sections = data.series.map(
        (s) => (s.config[key] ?? {}) as Record<string, unknown>,
      );
      return [...new Set(sections.flatMap((s) => Object.keys(s)))]
        .filter((field) =>
          sections.some(
            (s) =>
              JSON.stringify(s[field]) !== JSON.stringify(sections[0][field]),
          ),
        )
        .map((field) => ({
          key: `${key}.${field}`,
          title: label(field),
          values: sections.map((s) => s[field]),
        }));
    }) ?? [];
  const points = new Map<number, Record<string, number>>();
  data?.series.forEach((series, index) =>
    series.curve.rows.forEach((row) => {
      const t = Date.parse(row.time);
      points.set(t, {
        ...points.get(t),
        timestamp: t,
        [`equity${index}`]: row.equity,
        [`drawdown${index}`]: row.drawdown,
      });
    }),
  );
  const rows = [...points.values()].sort((a, b) => a.timestamp - b.timestamp);
  return (
    <>
      <h2>Compare saved hypotheses</h2>
      <p className="muted">
        Configuration differences first. Each series retains its original dates
        and full-history statistics; no automatic winner selection.
      </p>
      <Status {...state} />
      {data && (
        <>
          {data.warnings.map((w) => (
            <p className="notice" key={w}>
              {w}
            </p>
          ))}
          <section className="panel">
            <h2>Configuration differences</h2>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Setting</th>
                    {data.series.map((s) => (
                      <th key={s.id}>{s.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {differences.map(({ key, title, values }) => (
                    <tr key={key}>
                      <th scope="row">{title}</th>
                      {data.series.map((s, index) => (
                        <td key={s.id}>
                          <code>{JSON.stringify(values[index])}</code>
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {!data.differences.length && (
              <p className="muted">
                Same strategy configuration; inspect data versions and evidence.
              </p>
            )}
          </section>
          <section className="panel">
            <h2>Portfolio and universe comparison</h2>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Metric</th>
                    {data.series.map((s) => (
                      <th key={s.id}>{s.name}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    "total_return",
                    "benchmark_excess_return",
                    "sharpe",
                    "sortino",
                    "max_drawdown",
                    "fee_drag",
                    "funding_drag",
                    "turnover",
                  ].map((key) => (
                    <tr key={key}>
                      <th>{label(key)}</th>
                      {data.series.map((s) => (
                        <td
                          key={s.id}
                          title={s.analytics.metrics[key]?.reason ?? ""}
                        >
                          {formatValue(
                            s.analytics.metrics[key]?.value,
                            s.analytics.metrics[key]?.unit ?? "ratio",
                          )}
                        </td>
                      ))}
                    </tr>
                  ))}
                  <tr>
                    <th>Mean trader membership turnover</th>
                    {data.series.map((s) => (
                      <td key={s.id}>
                        {formatValue(s.membership_turnover, "percent")}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <th>Mean market membership turnover</th>
                    {data.series.map((s) => (
                      <td key={s.id}>
                        {formatValue(s.market_membership_turnover, "percent")}
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <th>Mean selected markets</th>
                    {data.series.map((s) => (
                      <td key={s.id}>
                        {formatValue(s.mean_selected_assets, "ratio")}
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
          <div className="toolbar">
            <label>
              Equity display
              <select
                aria-label="Equity display"
                value={units}
                onChange={(e) => setUnits(e.target.value)}
              >
                <option value="growth">Growth of one (backend rebased)</option>
                <option value="usd">Dollar equity (original capital)</option>
              </select>
            </label>
          </div>
          <Plot
            title="Saved backtest equity"
            rows={rows}
            unit={units === "growth" ? "ratio" : "usd"}
            lines={data.series.map((s, i) => [`equity${i}`, s.name, colors[i]])}
          />
          <Plot
            title="Saved backtest drawdown"
            rows={rows}
            unit="percent"
            lines={data.series.map((s, i) => [
              `drawdown${i}`,
              s.name,
              colors[i],
            ])}
          />
        </>
      )}
    </>
  );
}
