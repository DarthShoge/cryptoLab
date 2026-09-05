import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { EquityPage } from "./api";
import { formatValue } from "./format";

function Plot({
  title,
  rows,
  lines,
  unit,
}: {
  title: string;
  rows: object[];
  lines: [string, string, string][];
  unit: string;
}) {
  return (
    <section className="panel chart-panel">
      <h3>{title}</h3>
      <div
        className="chart"
        role="img"
        aria-label={`${title}; UTC timestamps. Values available in the summary and downloadable equity table.`}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={rows}
            margin={{ top: 12, right: 15, bottom: 4, left: 12 }}
          >
            <CartesianGrid
              stroke="#273540"
              strokeDasharray="3 5"
              vertical={false}
            />
            <XAxis
              dataKey="timestamp"
              type="number"
              scale="time"
              domain={["dataMin", "dataMax"]}
              tickFormatter={(v) => new Date(v).toISOString().slice(11, 19)}
              tick={{ fill: "#9aabb5", fontSize: 11 }}
              minTickGap={40}
            />
            <YAxis
              domain={["auto", "auto"]}
              tickFormatter={(v) =>
                unit === "percent"
                  ? `${(v * 100).toFixed(1)}%`
                  : Number(v).toLocaleString("en-US", {
                      maximumFractionDigits: 0,
                    })
              }
              tick={{ fill: "#9aabb5", fontSize: 11 }}
              width={65}
            />
            <Tooltip
              contentStyle={{
                background: "#15212a",
                border: "1px solid #40535f",
                borderRadius: 8,
              }}
              labelFormatter={(v) =>
                new Date(Number(v)).toISOString().replace("T", " ")
              }
              formatter={(v: number) => formatValue(v, unit)}
            />
            <Legend />
            {lines.map(([key, name, color]) => (
              <Line
                key={key}
                type="linear"
                dataKey={key}
                name={name}
                stroke={color}
                strokeWidth={2}
                dot={false}
                connectNulls
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
export function Charts({
  curve,
  benchmark,
}: {
  curve: EquityPage;
  benchmark?: EquityPage;
}) {
  if (!curve.available || !curve.rows.length)
    return <p className="empty">{curve.reason ?? "No equity observations"}</p>;
  const aligned =
    benchmark?.available &&
    benchmark.rows[0]?.time === curve.rows[0]?.time &&
    benchmark.rows.at(-1)?.time === curve.rows.at(-1)?.time;
  const points = new Map<number, Record<string, number>>();
  for (const row of curve.rows) {
    const t = Date.parse(row.time);
    points.set(t, {
      timestamp: t,
      equity: row.equity,
      drawdown: row.drawdown,
      gross: row.gross_exposure,
      net: row.net_exposure,
    });
  }
  if (aligned)
    for (const row of benchmark!.rows) {
      const t = Date.parse(row.time);
      points.set(t, {
        ...points.get(t),
        timestamp: t,
        benchmark: row.equity,
        benchmarkDrawdown: row.drawdown,
      });
    }
  const rows = [...points.values()].sort((a, b) => a.timestamp - b.timestamp);
  return (
    <>
      <div className="chart-caption">
        {curve.total.toLocaleString()} minute observations · UTC ·{" "}
        {curve.downsampled
          ? "extrema-preserving display sampling"
          : "full-resolution display"}
        {benchmark && !aligned
          ? " · benchmark window unavailable or mismatched"
          : ""}
      </div>
      <Plot
        title="Equity vs benchmark"
        rows={rows}
        unit="usd"
        lines={[
          ["equity", "Strategy", "#7fe2ca"],
          ...(aligned
            ? [
                ["benchmark", "Benchmark", "#8b9de8"] as [
                  string,
                  string,
                  string,
                ],
              ]
            : []),
        ]}
      />
      <div className="chart-grid">
        <Plot
          title="Underwater · drawdown"
          rows={rows}
          unit="percent"
          lines={[
            ["drawdown", "Drawdown", "#f2ab89"],
            ...(aligned
              ? [
                  ["benchmarkDrawdown", "Benchmark", "#8b9de8"] as [
                    string,
                    string,
                    string,
                  ],
                ]
              : []),
          ]}
        />
        <Plot
          title="Gross & net exposure"
          rows={rows}
          unit="usd"
          lines={[
            ["gross", "Gross", "#90a6ed"],
            ["net", "Net", "#7fe2ca"],
          ]}
        />
      </div>
    </>
  );
}
