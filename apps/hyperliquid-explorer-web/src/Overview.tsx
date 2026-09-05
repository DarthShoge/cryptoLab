import { useState } from "react";
import {
  type Analytics,
  type Detail,
  type EquityPage,
  type Metric,
  type Scenario,
  runUrl,
  scenarioParams,
  useResource,
} from "./api";
import {
  compareNullable,
  formatValue,
  label,
  scenarioKey,
  utc,
} from "./format";
import { Status } from "./Status";
import { Charts } from "./Charts";

export function ScenarioSelect({
  scenarios,
  selected,
  onChange,
}: {
  scenarios: Scenario[];
  selected: Scenario;
  onChange: (s: Scenario) => void;
}) {
  const names = [
    ...new Set(scenarios.map((s) => `${s.scenario_type}:${s.name}`)),
  ];
  return (
    <>
      <label>
        Scenario
        <select
          value={`${selected.scenario_type}:${selected.name}`}
          onChange={(e) =>
            onChange(
              scenarios.find(
                (s) =>
                  `${s.scenario_type}:${s.name}` === e.target.value &&
                  s.latency_seconds === selected.latency_seconds,
              ) ??
                scenarios.find(
                  (s) => `${s.scenario_type}:${s.name}` === e.target.value,
                )!,
            )
          }
        >
          {names.map((n) => (
            <option key={n} value={n}>
              {label(n.split(":")[1])}
              {n.startsWith("control:") ? " · benchmark" : ""}
            </option>
          ))}
        </select>
      </label>
      <label>
        Execution delay
        <select
          disabled={selected.latency_seconds === null}
          value={selected.latency_seconds ?? "cash"}
          onChange={(e) =>
            onChange(
              scenarios.find(
                (s) =>
                  s.name === selected.name &&
                  s.scenario_type === selected.scenario_type &&
                  s.latency_seconds === Number(e.target.value),
              )!,
            )
          }
        >
          {selected.latency_seconds === null ? (
            <option value="cash">Not applicable</option>
          ) : (
            scenarios
              .filter(
                (s) =>
                  s.name === selected.name &&
                  s.scenario_type === selected.scenario_type,
              )
              .map((s) => (
                <option key={s.latency_seconds} value={s.latency_seconds!}>
                  {s.latency_seconds} seconds
                </option>
              ))
          )}
        </select>
      </label>
    </>
  );
}
function Value({ metric }: { metric?: Metric }) {
  return (
    <>
      <strong>{formatValue(metric?.value, metric?.unit ?? "ratio")}</strong>
      {metric?.reason && <small>{metric.reason}</small>}
    </>
  );
}

export function Overview({
  detail,
  selected,
  onChange,
}: {
  detail: Detail;
  selected: Scenario;
  onChange: (s: Scenario) => void;
}) {
  const [benchmarkKey, setBenchmarkKey] = useState(
    scenarioKey(
      detail.scenarios.find((s) => s.name === "cash") ?? detail.scenarios[0],
    ),
  );
  const benchmark =
    detail.scenarios.find((s) => scenarioKey(s) === benchmarkKey) ??
    detail.scenarios[0];
  const params = scenarioParams(selected);
  params.set("benchmark_type", benchmark.scenario_type);
  params.set("benchmark_name", benchmark.name);
  if (benchmark.latency_seconds !== null)
    params.set("benchmark_latency", String(benchmark.latency_seconds));
  const analysis = useResource<Analytics>(
    runUrl(detail.id, `/analytics?${params}`),
  );
  const curve = useResource<EquityPage>(
    runUrl(detail.id, `/equity?${scenarioParams(selected)}`),
  );
  const other = useResource<EquityPage>(
    runUrl(detail.id, `/equity?${scenarioParams(benchmark)}`),
  );
  const [sort, setSort] = useState("total_return"),
    [descending, setDescending] = useState(true);
  const [columns, setColumns] = useState([
    "total_return",
    "sharpe",
    "sortino",
    "max_drawdown",
    "fee_drag",
    "residual_gross_exposure",
  ]);
  const metrics = analysis.data?.metrics;
  const groups = [...new Set(Object.values(metrics ?? {}).map((m) => m.group))];
  const storedColumns = Object.keys(metrics ?? {}).filter((k) =>
    detail.scenarios.some((s) => k in s.metrics),
  );
  return (
    <>
      <div className="toolbar">
        <ScenarioSelect
          scenarios={detail.scenarios}
          selected={selected}
          onChange={onChange}
        />
        <label>
          Compare against
          <select
            value={benchmarkKey}
            onChange={(e) => setBenchmarkKey(e.target.value)}
          >
            {detail.scenarios
              .filter((s) => s.scenario_type === "control")
              .map((s) => (
                <option key={scenarioKey(s)} value={scenarioKey(s)}>
                  {label(s.name)}
                  {s.latency_seconds !== null ? ` · ${s.latency_seconds}s` : ""}
                </option>
              ))}
          </select>
        </label>
      </div>
      <Status {...analysis} />
      {metrics && (
        <>
          <div className="metric-cards">
            {[
              "final_equity",
              "total_return",
              "sharpe",
              "sortino",
              "max_drawdown",
            ].map((key) => (
              <article
                className="metric-card"
                key={key}
                title={metrics[key]?.description}
              >
                <span>{label(key)}</span>
                <Value metric={metrics[key]} />
              </article>
            ))}
          </div>
          <details className="panel analysis">
            <summary>
              Portfolio analysis <span>Return, risk, costs & exposure</span>
            </summary>
            <p className="muted">{analysis.data?.conventions}</p>
            <p className="muted">
              {utc(metrics.total_return?.start)} →{" "}
              {utc(metrics.total_return?.end)} ·{" "}
              {metrics.total_return?.samples.toLocaleString()} observations
            </p>
            {analysis.data?.warnings
              .filter((w) => !detail.warnings?.includes(w))
              .map((w) => (
                <p className="notice" key={w}>
                  {w}
                </p>
              ))}
            <div className="analysis-grid">
              {groups.map((group) => (
                <section key={group}>
                  <h3>{group}</h3>
                  <table>
                    <thead>
                      <tr>
                        <th>Metric</th>
                        <th>Selected</th>
                        <th>Benchmark</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(metrics)
                        .filter(([, m]) => m.group === group)
                        .map(([key, m]) => (
                          <tr key={key}>
                            <th scope="row">
                              <span title={m.description}>
                                {label(key)}{" "}
                                <span
                                  aria-label={m.description}
                                  className="hint"
                                >
                                  ⓘ
                                </span>
                              </span>
                            </th>
                            <td>
                              <Value metric={m} />
                            </td>
                            <td>
                              <Value metric={analysis.data?.benchmark?.[key]} />
                            </td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </section>
              ))}
            </div>
          </details>
        </>
      )}
      <Status {...curve} />
      <Status {...other} />
      {curve.data && <Charts curve={curve.data} benchmark={other.data} />}
      <section className="panel">
        <div className="section-heading">
          <div>
            <h2>Scenario comparison</h2>
            <p className="muted">
              Same report window · click a heading to sort · unavailable metrics
              last
            </p>
          </div>
          <span className="pill">{detail.scenarios.length} scenarios</span>
        </div>
        <details className="column-picker">
          <summary>Choose metric columns</summary>
          <p className="muted">
            Stored scenario metrics; additional derived statistics are in
            Portfolio analysis. Annualized values are unavailable for synthetic
            or under-one-day histories.
          </p>
          <div>
            {storedColumns.map((key) => (
              <label key={key}>
                <input
                  type="checkbox"
                  checked={columns.includes(key)}
                  onChange={(e) =>
                    setColumns(
                      e.target.checked
                        ? [...columns, key]
                        : columns.filter((c) => c !== key),
                    )
                  }
                />
                {label(key)}
              </label>
            ))}
          </div>
        </details>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>Scenario</th>
                <th>Delay</th>
                {columns.map((key) => (
                  <th key={key}>
                    <button
                      className="sort"
                      onClick={() => {
                        setDescending(sort === key ? !descending : true);
                        setSort(key);
                      }}
                    >
                      {label(key)}{" "}
                      {sort === key ? (descending ? "↓" : "↑") : ""}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[...detail.scenarios]
                .sort((a, b) =>
                  compareNullable(a.metrics[sort], b.metrics[sort], descending),
                )
                .map((s) => (
                  <tr
                    key={scenarioKey(s)}
                    className={
                      scenarioKey(s) === scenarioKey(selected)
                        ? "selected-row"
                        : ""
                    }
                  >
                    <th>
                      <button
                        className="link-button"
                        onClick={() => onChange(s)}
                      >
                        {label(s.name)}
                      </button>
                      <small>{s.scenario_type}</small>
                    </th>
                    <td>
                      {s.latency_seconds === null
                        ? "—"
                        : `${s.latency_seconds}s`}
                    </td>
                    {columns.map((k) => (
                      <td key={k}>
                        {formatValue(
                          s.metrics[k],
                          metrics?.[k]?.unit ?? "ratio",
                        )}
                      </td>
                    ))}
                  </tr>
                ))}
            </tbody>
          </table>
        </div>
      </section>
    </>
  );
}
