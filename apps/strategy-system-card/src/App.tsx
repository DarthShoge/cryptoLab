import { useMemo, useState } from "react";

import { purePerpSignalSystemCardData as purePerpData } from "./data/purePerpSignalSystemCardData";
import type { AssetDiagnosticChart, AssetDiagnosticPoint } from "./data/purePerpSignalSystemCardData";
import { strategySystemCardData as kaminoData } from "./data/strategySystemCardData";
import type { ChartPoint, MetricRow, StrategySystemCardData, TrafficState } from "./types";

const sections = [
  ["identity", "System Identity"],
  ["verdict", "Executive Verdict"],
  ["mechanics", "Strategy Mechanics"],
  ["traffic", "Traffic-Light Engine"],
  ["risk", "Risk Governors"],
  ["evidence", "Evidence"],
  ["regimes", "Regime Playbook"],
  ["transfer", "Scenario Transfer"],
  ["usage", "Use / Do Not Use"],
  ["failures", "Failure Modes"],
  ["production", "Production Readiness"],
  ["audit", "Audit Trail"],
] as const;

const colors = {
  teal: "#0f766e",
  blue: "#1d4ed8",
  amber: "#b45309",
  red: "#b91c1c",
  green: "#047857",
  purple: "#7c3aed",
};

type Series = {
  name: string;
  points: ChartPoint[];
  color: string;
  width?: number;
  opacity?: number;
};

type Column = {
  key: string;
  label: string;
  render?: (row: MetricRow) => string;
};

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

function fmtMoney(value: unknown): string {
  const number = asNumber(value);
  if (number === null) return "-";
  return number.toLocaleString(undefined, {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });
}

function fmt(value: unknown, digits = 2): string {
  const number = asNumber(value);
  if (number === null) return "-";
  return number.toLocaleString(undefined, {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  });
}

function fmtCompact(value: unknown): string {
  const number = asNumber(value);
  if (number === null) return "-";
  return Intl.NumberFormat(undefined, {
    notation: "compact",
    maximumFractionDigits: 2,
  }).format(number);
}

function pct(value: unknown, digits = 2): string {
  const number = asNumber(value);
  return number === null ? "-" : `${fmt(number, digits)}%`;
}

function MetricCard({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {note ? <div className="metric-note">{note}</div> : null}
    </div>
  );
}

function Section({
  id,
  title,
  subtitle,
  children,
}: {
  id: string;
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="section">
      <div className="section-header">
        <p className="eyebrow">System Card</p>
        <h2>{title}</h2>
        <p className="subtitle">{subtitle}</p>
      </div>
      {children}
    </section>
  );
}

function Callout({
  tone,
  title,
  children,
}: {
  tone: "good" | "warning" | "danger";
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`callout ${tone}`}>
      <h3>{title}</h3>
      <p>{children}</p>
    </div>
  );
}

function Checklist({ items, negative = false }: { items: string[]; negative?: boolean }) {
  return (
    <ul className="check-list">
      {items.map((item) => (
        <li key={item}>
          <span className={negative ? "xmark" : "check"}>{negative ? "!" : "✓"}</span>
          <span>{item}</span>
        </li>
      ))}
    </ul>
  );
}

function DataTable({ rows, columns }: { rows: MetricRow[]; columns: Column[] }) {
  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((column) => (
              <th key={column.key}>{column.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={`${asString(row.name)}-${rowIndex}`}>
              {columns.map((column) => (
                <td key={column.key}>{column.render ? column.render(row) : asString(row[column.key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ChartCard({
  title,
  subtitle,
  children,
  legend,
}: {
  title: string;
  subtitle?: string;
  children: React.ReactNode;
  legend?: { label: string; color: string }[];
}) {
  return (
    <div className="chart-card">
      <div className="chart-title">
        <h3>{title}</h3>
        <span>{subtitle}</span>
      </div>
      <div className="chart-wrap">{children}</div>
      {legend ? (
        <div className="legend">
          {legend.map((item) => (
            <span className="legend-item" key={item.label} style={{ color: item.color }}>
              <span className="dot" />
              {item.label}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function makePath(
  points: ChartPoint[],
  yField: keyof ChartPoint,
  width: number,
  height: number,
  padding: { top: number; right: number; bottom: number; left: number },
  yMin: number,
  yMax: number,
): string {
  const valid = points.filter((point) => asNumber(point[yField]) !== null);
  if (valid.length < 2) return "";
  const denom = Math.max(yMax - yMin, 1e-9);
  return valid
    .map((point, index) => {
      const value = asNumber(point[yField]) ?? 0;
      const x = padding.left + (index / Math.max(valid.length - 1, 1)) * (width - padding.left - padding.right);
      const y = padding.top + (1 - (value - yMin) / denom) * (height - padding.top - padding.bottom);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function MultiLineChart({ series, yField = "normalized_value" }: { series: Series[]; yField?: keyof ChartPoint }) {
  const width = 920;
  const height = 290;
  const padding = { top: 18, right: 26, bottom: 34, left: 54 };
  const values = series.flatMap((item) =>
    item.points.map((point) => asNumber(point[yField])).filter((value): value is number => value !== null),
  );
  const yMin = Math.min(...values, 0);
  const yMax = Math.max(...values, 100);
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((ratio) => yMin + (yMax - yMin) * ratio);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Line chart">
      <rect x="0" y="0" width={width} height={height} fill="#fbfcfe" />
      {ticks.map((tick) => {
        const y = padding.top + (1 - (tick - yMin) / Math.max(yMax - yMin, 1e-9)) * (height - padding.top - padding.bottom);
        return (
          <g key={tick}>
            <line x1={padding.left} x2={width - padding.right} y1={y} y2={y} stroke="#e2e8f0" />
            <text x="8" y={y + 4} fill="#64748b" fontSize="11">
              {fmt(tick, 0)}
            </text>
          </g>
        );
      })}
      {series.map((item) => (
        <path
          key={item.name}
          d={makePath(item.points, yField, width, height, padding, yMin, yMax)}
          fill="none"
          stroke={item.color}
          strokeWidth={item.width ?? 2.4}
          opacity={item.opacity ?? 1}
        />
      ))}
      <text x={padding.left} y={height - 8} fill="#64748b" fontSize="11">
        2021
      </text>
      <text x={width - padding.right - 34} y={height - 8} fill="#64748b" fontSize="11">
        2026
      </text>
    </svg>
  );
}

function DrawdownChart({ points }: { points: ChartPoint[] }) {
  const width = 920;
  const height = 260;
  const padding = { top: 18, right: 26, bottom: 34, left: 54 };
  const yMin = -100;
  const yMax = 0;
  const path = makePath(points, "drawdown_pct", width, height, padding, yMin, yMax);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Drawdown chart">
      <rect x="0" y="0" width={width} height={height} fill="#fbfcfe" />
      {[-100, -75, -50, -25, 0].map((tick) => {
        const y = padding.top + (1 - (tick - yMin) / (yMax - yMin)) * (height - padding.top - padding.bottom);
        return (
          <g key={tick}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={y}
              y2={y}
              stroke={tick === -50 ? "#f59e0b" : "#e2e8f0"}
              strokeDasharray={tick === -50 ? "7 5" : "0"}
            />
            <text x="8" y={y + 4} fill="#64748b" fontSize="11">
              {tick}%
            </text>
          </g>
        );
      })}
      <path d={path} fill="none" stroke={colors.red} strokeWidth="2.4" />
      <text x={width - padding.right - 150} y="130" fill={colors.amber} fontSize="12">
        50% investor tolerance line
      </text>
    </svg>
  );
}

function assetDiagnosticPath(
  points: AssetDiagnosticPoint[],
  key: keyof Pick<AssetDiagnosticPoint, "strategy" | "buyHold" | "relativePct" | "drawdownPct">,
  xForDate: (timestamp: string) => number,
  yForValue: (value: number) => number,
  logScale = false,
): string {
  return points
    .map((point, index) => {
      const raw = point[key];
      const value = logScale ? Math.log(Math.max(raw, 0.01)) : raw;
      return `${index === 0 ? "M" : "L"} ${xForDate(point.timestamp).toFixed(2)} ${yForValue(value).toFixed(2)}`;
    })
    .join(" ");
}

function yScale(value: number, min: number, max: number, top: number, height: number): number {
  return top + (1 - (value - min) / Math.max(max - min, 1e-9)) * height;
}

function dateNumber(timestamp: string): number {
  return new Date(`${timestamp}T00:00:00Z`).getTime();
}

function AssetDiagnosticChartCard({ chart }: { chart: AssetDiagnosticChart }) {
  const width = 980;
  const height = 720;
  const left = 72;
  const right = 28;
  const plotWidth = width - left - right;
  const topPanel = { top: 86, height: 260 };
  const relPanel = { top: 404, height: 130 };
  const ddPanel = { top: 586, height: 92 };
  const firstDate = dateNumber(chart.points[0].timestamp);
  const lastDate = dateNumber(chart.points[chart.points.length - 1].timestamp);
  const xForDate = (timestamp: string) => left + ((dateNumber(timestamp) - firstDate) / Math.max(lastDate - firstDate, 1)) * plotWidth;
  const shadeStart = Math.max(left, xForDate("2024-10-01"));
  const shadeEnd = Math.min(left + plotWidth, xForDate("2025-12-31"));

  const growthValues = chart.points.flatMap((point) => [point.strategy, point.buyHold]).map((value) => Math.log(Math.max(value, 0.01)));
  const growthMin = Math.min(...growthValues) * 0.98;
  const growthMax = Math.max(...growthValues) * 1.02;
  const relValues = chart.points.map((point) => point.relativePct).concat(chart.worstRelativePct);
  const relMin = Math.min(...relValues, 0) - 12;
  const relMax = Math.max(...relValues, 0) + 12;
  const ddMin = Math.min(...chart.points.map((point) => point.drawdownPct), -chart.maxDrawdownPct) - 4;
  const ddMax = 4;
  const growthY = (value: number) => yScale(value, growthMin, growthMax, topPanel.top, topPanel.height);
  const relY = (value: number) => yScale(value, relMin, relMax, relPanel.top, relPanel.height);
  const ddY = (value: number) => yScale(value, ddMin, ddMax, ddPanel.top, ddPanel.height);
  const worstX = xForDate(chart.worstRelativeDate);
  const maxDdX = xForDate(chart.maxDrawdownDate);

  return (
    <div className="chart-card asset-diagnostic-card">
      <div className="chart-title">
        <h3>{chart.title}</h3>
        <span>{chart.subtitle}</span>
      </div>
      <div className="chart-wrap">
        <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${chart.asset} profile diagnostic chart`}>
          <rect width={width} height={height} fill="#ffffff" />
          <text x={left} y="30" className="svg-title">
            {chart.title}
          </text>
          <text x={left} y="52" className="svg-subtitle">
            Strategy {pct(chart.totalReturnPct, 1)} | {chart.asset} B&H {pct(chart.buyHoldPct, 1)} | Max DD {pct(chart.maxDrawdownPct, 1)} | Worst relative {pct(chart.worstRelativePct, 1)}
          </text>

          {[topPanel, relPanel, ddPanel].map((panel, index) => (
            <g key={index}>
              <rect x={left} y={panel.top} width={plotWidth} height={panel.height} fill="#fbfdff" stroke="#d8dee8" />
              <rect x={shadeStart} y={panel.top} width={Math.max(shadeEnd - shadeStart, 0)} height={panel.height} fill="#c23030" opacity="0.07" />
            </g>
          ))}

          <text className="axis-label" x="18" y={topPanel.top + topPanel.height / 2} transform={`rotate(-90 18 ${topPanel.top + topPanel.height / 2})`}>
            Growth of $1, log scale
          </text>
          <text className="axis-label" x="18" y={relPanel.top + relPanel.height / 2} transform={`rotate(-90 18 ${relPanel.top + relPanel.height / 2})`}>
            Relative to B&H
          </text>
          <text className="axis-label" x="18" y={ddPanel.top + ddPanel.height / 2} transform={`rotate(-90 18 ${ddPanel.top + ddPanel.height / 2})`}>
            Drawdown
          </text>

          {[Math.exp(growthMin), 100, Math.exp(growthMax)].map((tick) => {
            const y = growthY(Math.log(Math.max(tick, 0.01)));
            return (
              <g key={`growth-${tick}`}>
                <line x1={left} x2={left + plotWidth} y1={y} y2={y} stroke="#e2e8f0" />
                <text x="28" y={y + 4} className="svg-label">
                  {`${fmt(tick / 100, 1)}x`}
                </text>
              </g>
            );
          })}
          {[relMin, 0, relMax].map((tick) => {
            const y = relY(tick);
            return (
              <g key={`rel-${tick}`}>
                <line x1={left} x2={left + plotWidth} y1={y} y2={y} stroke={tick === 0 ? "#52606d" : "#e2e8f0"} />
                <text x="26" y={y + 4} className="svg-label">
                  {pct(tick, 0)}
                </text>
              </g>
            );
          })}
          {[ddMin, -chart.maxDrawdownPct, 0].map((tick) => {
            const y = ddY(tick);
            return (
              <g key={`dd-${tick}`}>
                <line x1={left} x2={left + plotWidth} y1={y} y2={y} stroke={tick === -chart.maxDrawdownPct ? "#f59e0b" : "#e2e8f0"} strokeDasharray={tick === -chart.maxDrawdownPct ? "7 5" : "0"} />
                <text x="24" y={y + 4} className="svg-label">
                  {pct(tick, 0)}
                </text>
              </g>
            );
          })}

          <path d={assetDiagnosticPath(chart.points, "strategy", xForDate, growthY, true)} className="strategy-line" />
          <path d={assetDiagnosticPath(chart.points, "buyHold", xForDate, growthY, true)} className="buyhold-line" />
          <path d={assetDiagnosticPath(chart.points, "relativePct", xForDate, relY)} className="relative-line" />
          <path d={assetDiagnosticPath(chart.points, "drawdownPct", xForDate, ddY)} className="drawdown-line" />

          <circle cx={worstX} cy={relY(chart.worstRelativePct)} r="4.5" fill="#c23030" />
          <text x={Math.min(worstX + 10, left + plotWidth - 260)} y={relY(chart.worstRelativePct) - 8} className="svg-note">
            Worst relative: {pct(chart.worstRelativePct, 1)} on {chart.worstRelativeDate}
          </text>
          <circle cx={maxDdX} cy={ddY(-chart.maxDrawdownPct)} r="4.5" fill="#0b5cad" />
          <text x={Math.min(maxDdX + 10, left + plotWidth - 220)} y={ddY(-chart.maxDrawdownPct) - 8} className="svg-label">
            Max DD: -{pct(chart.maxDrawdownPct, 1)}
          </text>
          <text x={shadeStart + 8} y={topPanel.top + 18} className="svg-note">
            Q4 2024-2025 focus window
          </text>

          <g transform={`translate(${left + plotWidth - 260}, ${topPanel.top + 18})`}>
            <line x1="0" x2="18" y1="0" y2="0" className="strategy-line" />
            <text x="26" y="4" className="svg-label">Strategy</text>
            <line x1="0" x2="18" y1="20" y2="20" className="buyhold-line" />
            <text x="26" y="24" className="svg-label">{chart.asset} buy-and-hold</text>
            <line x1="0" x2="18" y1="40" y2="40" className="relative-line" />
            <text x="26" y="44" className="svg-label">Relative</text>
          </g>

          {["2022-01", "2023-01", "2024-01", "2025-01", "2026-01", "2026-07"].map((label) => {
            const x = xForDate(`${label}-01`);
            return (
              <g key={label}>
                <line x1={x} x2={x} y1={ddPanel.top + ddPanel.height} y2={ddPanel.top + ddPanel.height + 6} stroke="#d8dee8" />
                <text x={x - 20} y={ddPanel.top + ddPanel.height + 24} className="svg-label">
                  {label}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
}

function BarChart({ rows, valueKey, color }: { rows: MetricRow[]; valueKey: string; color: string }) {
  const width = 920;
  const height = 320;
  const padding = { top: 20, right: 22, bottom: 82, left: 60 };
  const values = rows.map((row) => asNumber(row[valueKey]) ?? 0);
  const max = Math.max(...values, 1);
  const barGap = 12;
  const barWidth = (width - padding.left - padding.right - barGap * (rows.length - 1)) / rows.length;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={valueKey}>
      <rect x="0" y="0" width={width} height={height} fill="#fbfcfe" />
      {rows.map((row, index) => {
        const value = asNumber(row[valueKey]) ?? 0;
        const barHeight = (value / max) * (height - padding.top - padding.bottom);
        const x = padding.left + index * (barWidth + barGap);
        const y = height - padding.bottom - barHeight;
        const label = asString(row.name).replace("best_mechanics_", "").replace("_directional", "").replace("control_best_", "");
        return (
          <g key={`${label}-${valueKey}`}>
            <rect x={x} y={y} width={barWidth} height={barHeight} rx="5" fill={color} opacity="0.86" />
            <text x={x + barWidth / 2} y={y - 8} textAnchor="middle" fill="#334155" fontSize="11" fontWeight="700">
              {valueKey.includes("drawdown") ? pct(value, 1) : fmtCompact(value)}
            </text>
            <text x={x + barWidth / 2} y={height - padding.bottom + 18} textAnchor="middle" fill="#64748b" fontSize="10">
              {label.slice(0, 18)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function CardSelector({ selected, onSelect }: { selected: "kamino" | "pure-perp"; onSelect: (card: "kamino" | "pure-perp") => void }) {
  return (
    <div className="card-selector" aria-label="System card selector">
      <button className={selected === "kamino" ? "active" : ""} onClick={() => onSelect("kamino")} type="button">
        Kamino Governor
      </button>
      <button className={selected === "pure-perp" ? "active" : ""} onClick={() => onSelect("pure-perp")} type="button">
        Pure Perp Signal
      </button>
    </div>
  );
}

function TrafficStateCards({ states }: { states: TrafficState[] }) {
  const palette = ["#047857", "#ca8a04", "#ea580c", "#dc2626", "#2563eb"];
  return (
    <div className="three-col">
      {states.map((state, index) => (
        <div className="info-card" key={state.state} style={{ borderTop: `4px solid ${palette[index]}` }}>
          <h3>{state.state}</h3>
          <p>{state.meaning}</p>
          <p className="spaced">{state.behavior}</p>
        </div>
      ))}
    </div>
  );
}

function Hero({ data }: { data: StrategySystemCardData }) {
  const top = data.topCandidate;
  return (
    <header className="hero">
      <div className="hero-grid">
        <div className="hero-copy">
          <p className="eyebrow">Strategy System Card</p>
          <h2>SOL/ETH Kamino Traffic-Light Governor</h2>
          <p className="hero-lede">
            A long-biased, path-dependent crypto leverage strategy that uses multi-timeframe traffic-light confirmation, SOL/ETH rotation,
            drawdown and volatility governors, recovery re-risking, and rebalance friction controls. The current top candidate is not a
            universal multi-asset model; its edge is strongly SOL-led.
          </p>
          <div className="pill-row">
            {["Research checkpoint", "Kamino-style mechanics", "Hourly path-dependent model", "Not production-ready without controls"].map((item) => (
              <span className="pill" key={item}>
                {item}
              </span>
            ))}
          </div>
        </div>
        <div className="hero-panel">
          <div className="metric-grid">
            <MetricCard label="Final USD" value={fmtMoney(top.final_portfolio_value_usd)} note="100 SOL initial collateral run" />
            <MetricCard label="Final SOL" value={fmt(top.final_sol_equiv, 3)} note="SOL-equivalent terminal value" />
            <MetricCard label="Annualized Return" value={pct(top.annualized_return_pct, 2)} note="CAGR from observed history" />
            <MetricCard label="Max Drawdown" value={pct(top.max_drawdown_pct, 2)} note="Still above 50% investor target" />
            <MetricCard label="Sortino" value={fmt(top.sortino_ratio, 3)} note="Best retained low-DD checkpoint" />
            <MetricCard label="Sharpe" value={fmt(top.sharpe_ratio_check, 3)} note="Hourly annualized check" />
            <MetricCard label="Actions / Year" value={fmt(top.action_turnover_per_year, 1)} note="After turnover controls" />
          </div>
        </div>
      </div>
    </header>
  );
}

function KaminoSystemCard({ selected, onSelect }: { selected: "kamino" | "pure-perp"; onSelect: (card: "kamino" | "pure-perp") => void }) {
  const data = kaminoData;
  const top = data.topCandidate;
  const scenarioRows = data.scenarioStrategies as MetricRow[];
  const topSeries = data.charts.topEquity;
  const scenarioSeries: Series[] = [
    { name: "SOL/ETH control", points: data.charts.controlEquity, color: colors.teal, width: 2.8 },
    { name: "SOL-only", points: data.charts.solOnlyEquity, color: colors.blue },
    { name: "ETH-only", points: data.charts.ethOnlyEquity, color: colors.purple },
    { name: "BTC-only", points: data.charts.btcOnlyEquity, color: colors.amber },
  ];
  const regimeRows = (data.regimes as MetricRow[]).filter((row) =>
    ["control_best_SOL_ETH", "best_mechanics_SOL_only_directional", "best_mechanics_BTC_only_directional", "best_mechanics_ETH_only_directional"].includes(
      asString(row.name),
    ),
  );

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-kicker">System Card</div>
          <h1>Kamino Traffic-Light Governor</h1>
          <p>{data.meta.window}</p>
        </div>
        <CardSelector selected={selected} onSelect={onSelect} />
        <nav className="nav-list">
          {sections.map(([id, label]) => (
            <a href={`#${id}`} key={id}>
              {label}
            </a>
          ))}
        </nav>
      </aside>

      <main className="main">
        <div className="content">
          <Hero data={data} />

          <Section id="identity" title="System Identity" subtitle="What this strategy is, what it optimizes, and what it should not be mistaken for.">
            <div className="two-col">
              <div className="info-card">
                <h3>Definition</h3>
                <p>
                  The top candidate is a SOL/ETH directional strategy for a Kamino-like lending account. It starts from SOL collateral in the
                  checkpoint run, borrows and repays USDC to manage long exposure, rotates between SOL and ETH when traffic-light ranking changes,
                  and uses state-dependent governors to avoid remaining maximally levered through deep drawdowns.
                </p>
              </div>
              <div className="info-card">
                <h3>Nature</h3>
                <p>
                  This is not a market-neutral system. It is a governed long-risk system. The research evidence says the edge is overwhelmingly
                  SOL-driven: SOL-only works well, BTC-only does not, and SOL/ETH is materially better than SOL-only in the retained tests.
                </p>
              </div>
            </div>
          </Section>

          <Section id="verdict" title="Executive Verdict" subtitle="The current checkpoint is attractive on return capture, but it remains too risky for mandates with a hard sub-50% drawdown constraint.">
            <div className="three-col">
              <Callout tone="good" title="What Works">
                The top candidate turns 100 SOL into 1158.856 SOL-equivalent while cutting raw SOL buy-and-hold drawdown from 96.80% to 56.88%.
              </Callout>
              <Callout tone="warning" title="What Still Fails">
                The drawdown remains above the stated 50% institutional tolerance. It is close, but not yet inside the target risk box.
              </Callout>
              <Callout tone="danger" title="What Not To Infer">
                Do not infer that the mechanics transfer to BTC. BTC-only underperformed BTC buy-and-hold despite lower drawdown.
              </Callout>
            </div>
          </Section>

          <Section id="mechanics" title="Strategy Mechanics" subtitle="The strategy is a layered decision system: signal ranking chooses the asset, governors choose exposure, and execution controls decide whether a rebalance is large enough to perform.">
            <div className="three-col">
              <div className="info-card">
                <h3>1. Signal Rank</h3>
                <p>Multi-timeframe supertrend signals are converted into green-count rankings. The long candidate is the strongest eligible asset among SOL and ETH.</p>
              </div>
              <div className="info-card">
                <h3>2. Exposure Target</h3>
                <p>Base long target is 1.075x, modified by drawdown tiers, realized volatility, and recovery state. The maximum observed target is 1.85x.</p>
              </div>
              <div className="info-card">
                <h3>3. Rebalance Gate</h3>
                <p>Green/yellow states use a 12-hour cooldown and 5% rebalance threshold. This reduces action count while retaining the core return profile.</p>
              </div>
            </div>
            <p className="footer-note">
              Current top candidate config: drawdown tiers at 30%, 42%, and 50%; realized-vol lookback 336 hours; recovery min drawdown 12%;
              recovery max worsening -2%; recovery min green 3.
            </p>
          </Section>

          <Section id="traffic" title="Traffic-Light Engine" subtitle="Traffic lights are the language of the model. They define whether the system can participate, should cool down, or should let risk governors dominate.">
            <TrafficStateCards states={data.trafficStates} />
          </Section>

          <Section id="risk" title="Risk Governors" subtitle="The model is built around accepting SOL-led upside while preventing raw buy-and-hold style collapse. These governors are the main defenses.">
            <div className="two-col">
              <div className="info-card">
                <h3>Governor Stack</h3>
                <Checklist items={data.governors} />
              </div>
              <div className="info-card">
                <h3>Risk Accounting</h3>
                <DataTable
                  rows={[top]}
                  columns={[
                    { key: "min_health_factor", label: "Min HF", render: (row) => fmt(row.min_health_factor, 3) },
                    { key: "bars_below_hf_1_5", label: "Bars HF < 1.5", render: (row) => fmt(row.bars_below_hf_1_5, 0) },
                    { key: "total_liquidations", label: "Liquidations", render: (row) => fmt(row.total_liquidations, 0) },
                    { key: "estimated_annualized_turnover_multiple", label: "Turnover / Yr", render: (row) => `${fmt(row.estimated_annualized_turnover_multiple, 2)}x` },
                  ]}
                />
              </div>
            </div>
          </Section>

          <Section id="evidence" title="Evidence and Charts" subtitle="Charts are sourced from retained local report artifacts and downsampled for static rendering.">
            <div className="two-col">
              <ChartCard title="Top Candidate Portfolio Path" subtitle="Indexed to 100" legend={[{ label: "Top candidate", color: colors.teal }]}>
                <MultiLineChart series={[{ name: "Top candidate", points: topSeries, color: colors.teal }]} />
              </ChartCard>
              <ChartCard title="Top Candidate Drawdown" subtitle="Negative drawdown percent" legend={[{ label: "Drawdown", color: colors.red }]}>
                <DrawdownChart points={topSeries} />
              </ChartCard>
            </div>
            <div className="chart-spacing">
              <ChartCard title="Scenario Equity Comparison" subtitle="Indexed to 100" legend={scenarioSeries.map((series) => ({ label: series.name, color: series.color }))}>
                <MultiLineChart series={scenarioSeries} />
              </ChartCard>
            </div>
          </Section>

          <Section id="regimes" title="Regime Playbook" subtitle="The system should be judged by how it behaves across regimes, not only by full-period terminal value.">
            <DataTable
              rows={regimeRows.filter((row) => ["full_2021_2026", "crash_2022", "post_2024", "ytd_2026"].includes(asString(row.regime)))}
              columns={[
                { key: "name", label: "Strategy", render: (row) => asString(row.name).replace("best_mechanics_", "").replace("_directional", "") },
                { key: "regime", label: "Regime" },
                { key: "return_pct", label: "Return", render: (row) => pct(row.return_pct, 1) },
                { key: "drawdown_pct", label: "DD", render: (row) => pct(row.drawdown_pct, 1) },
                { key: "avg_target_long_fraction", label: "Avg Long", render: (row) => fmt(row.avg_target_long_fraction, 3) },
                { key: "min_health_factor", label: "Min HF", render: (row) => fmt(row.min_health_factor, 3) },
              ]}
            />
          </Section>

          <Section id="transfer" title="Scenario Transfer Tests" subtitle="These tests ask whether the mechanics are generally good or whether the edge is asset-specific.">
            <div className="two-col">
              <ChartCard title="Final USD by Scenario" subtitle="$10k USDC transfer test">
                <BarChart rows={scenarioRows} valueKey="final_portfolio_value_usd" color={colors.teal} />
              </ChartCard>
              <ChartCard title="Max Drawdown by Scenario" subtitle="Lower is better">
                <BarChart rows={scenarioRows} valueKey="max_drawdown_pct" color={colors.red} />
              </ChartCard>
            </div>
            <div className="chart-spacing">
              <DataTable
                rows={scenarioRows}
                columns={[
                  { key: "name", label: "Scenario", render: (row) => asString(row.name).replace("best_mechanics_", "").replace("_directional", "") },
                  { key: "directional_symbols", label: "Assets" },
                  { key: "final_portfolio_value_usd", label: "Final USD", render: (row) => fmtMoney(row.final_portfolio_value_usd) },
                  { key: "annualized_return_pct", label: "Ann. Return", render: (row) => pct(row.annualized_return_pct, 2) },
                  { key: "max_drawdown_pct", label: "Max DD", render: (row) => pct(row.max_drawdown_pct, 2) },
                  { key: "sortino_ratio", label: "Sortino", render: (row) => fmt(row.sortino_ratio, 3) },
                  { key: "sharpe_ratio_check", label: "Sharpe", render: (row) => fmt(row.sharpe_ratio_check, 3) },
                ]}
              />
            </div>
            <div className="chart-spacing">
              <DataTable
                rows={data.benchmarks as MetricRow[]}
                columns={[
                  { key: "name", label: "Benchmark" },
                  { key: "final_portfolio_value_usd", label: "Final USD", render: (row) => fmtMoney(row.final_portfolio_value_usd) },
                  { key: "total_return_pct", label: "Total Return", render: (row) => pct(row.total_return_pct, 1) },
                  { key: "annualized_return_pct", label: "Ann. Return", render: (row) => pct(row.annualized_return_pct, 2) },
                  { key: "max_drawdown_pct", label: "Max DD", render: (row) => pct(row.max_drawdown_pct, 2) },
                  { key: "sharpe_ratio", label: "Sharpe", render: (row) => fmt(row.sharpe_ratio, 3) },
                ]}
              />
            </div>
            <p className="footer-note">Interpretation: SOL-only is strong, SOL/ETH is stronger, ETH-only is useful but smaller, and BTC-only is defensive but not return-competitive.</p>
          </Section>

          <Section id="usage" title="When To Use and When Not To Use" subtitle="The system card should be explicit about mandate fit. A high return is not a universal green light.">
            <div className="two-col">
              <div className="info-card">
                <h3>Use When</h3>
                <Checklist items={data.useCases} />
              </div>
              <div className="info-card">
                <h3>Do Not Use When</h3>
                <Checklist items={data.nonUseCases} negative />
              </div>
            </div>
          </Section>

          <Section id="failures" title="Failure Modes" subtitle="These are the known ways the strategy can disappoint or become unsafe.">
            <div className="three-col">
              {data.failureModes.map((item, index) => (
                <div className="callout warning" key={item}>
                  <h3>Failure {index + 1}</h3>
                  <p>{item}</p>
                </div>
              ))}
            </div>
          </Section>

          <Section id="production" title="Production Readiness" subtitle="Before live deployment, the research strategy needs operational controls around data, trading, borrow, health factor, and reconciliation.">
            <div className="two-col">
              <div className="info-card">
                <h3>Required Controls</h3>
                <Checklist items={data.productionControls} />
              </div>
              <div className="info-card">
                <h3>Current Readiness Judgment</h3>
                <p>
                  Research-grade, not production-ready. The model has a coherent mechanism and strong SOL-led evidence, but it still exceeds the
                  50% drawdown target and depends on live execution assumptions that are not yet modeled tightly enough.
                </p>
              </div>
            </div>
          </Section>

          <Section id="audit" title="Audit Trail" subtitle="The card is generated from local report artifacts so the displayed claims can be traced.">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Artifact</th>
                  </tr>
                </thead>
                <tbody>
                  {data.artifactLinks.map((item) => (
                    <tr key={item}>
                      <td>
                        <code>{item}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="footer-note">Generated typed data module: apps/strategy-system-card/src/data/strategySystemCardData.ts</p>
          </Section>
        </div>
      </main>
    </div>
  );
}

function PurePerpSystemCard({ selected, onSelect }: { selected: "kamino" | "pure-perp"; onSelect: (card: "kamino" | "pure-perp") => void }) {
  const data = purePerpData;
  const profileRows = data.profileMetrics as MetricRow[];
  const rotatingRows = data.rotatingPortfolioMetrics as MetricRow[];
  const profileSeries: Series[] = [
    { name: "BTC profile", points: data.charts.btc, color: colors.blue, width: 2.6 },
    { name: "ETH profile", points: data.charts.eth, color: colors.purple, width: 2.6 },
    { name: "SOL profile", points: data.charts.sol, color: colors.teal, width: 2.6 },
  ];
  const bestProfile = data.profileMetrics[2];

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-kicker">System Card</div>
          <h1>{data.meta.shortTitle}</h1>
          <p>{data.meta.window}</p>
        </div>
        <CardSelector selected={selected} onSelect={onSelect} />
        <nav className="nav-list">
          {sections.map(([id, label]) => (
            <a href={`#${id}`} key={id}>
              {label}
            </a>
          ))}
        </nav>
      </aside>

      <main className="main">
        <div className="content">
          <header className="hero">
            <div className="hero-grid">
              <div className="hero-copy">
                <p className="eyebrow">Strategy System Card</p>
                <h2>{data.meta.title}</h2>
                <p className="hero-lede">
                  A stateless, price-action signal family for crypto perpetual futures. The signal maps multi-timeframe trend state, volatility
                  targeting, short governors, and asset archetype settings into target exposure; exchange mechanics are handled downstream.
                </p>
                <div className="pill-row">
                  {["Pure signal layer", "Perp venue abstraction", "BTC / ETH / SOL profiles", "Universe design pending"].map((item) => (
                    <span className="pill" key={item}>
                      {item}
                    </span>
                  ))}
                </div>
              </div>
              <div className="hero-panel">
                <div className="metric-grid">
                  <MetricCard label="Best Profile" value="SOL" note={`${pct(bestProfile.total_return_pct, 2)} total return`} />
                  <MetricCard label="BTC Profile" value={pct(profileRows[0].total_return_pct, 2)} note="vs 25.55% buy-and-hold" />
                  <MetricCard label="ETH Profile" value={pct(profileRows[1].total_return_pct, 2)} note="vs -57.69% buy-and-hold" />
                  <MetricCard label="SOL Profile" value={pct(profileRows[2].total_return_pct, 2)} note="near long-only archetype" />
                  <MetricCard label="Best Rotation" value={pct(rotatingRows[0].total_return_pct, 2)} note="Selector A, but 52.13% DD" />
                  <MetricCard label="Status" value="Research" note="Checkpoint before universe design" />
                </div>
              </div>
            </div>
          </header>

          <Section id="identity" title="System Identity" subtitle="What this strategy is and what should remain outside the signal layer.">
            <div className="two-col">
              <div className="info-card">
                <h3>Definition</h3>
                <p>
                  The strategy is a pure perp exposure signal. It reads market data, computes closed-bar multi-timeframe indicators, and emits a
                  target exposure. It does not depend on previous account equity, realized PnL, or open-position history to decide the signal.
                </p>
              </div>
              <div className="info-card">
                <h3>Boundary</h3>
                <p>
                  Fees, funding, liquidation checks, rebalance deadbands, and venue-specific constraints are simulated outside the signal. This
                  keeps the signal portable across exchanges while still allowing realistic execution modeling.
                </p>
              </div>
            </div>
          </Section>

          <Section id="verdict" title="Executive Verdict" subtitle="The architecture transfers, but the current evidence rejects one universal parameter set.">
            <div className="three-col">
              <Callout tone="good" title="What Works">
                BTC, ETH, and SOL profiles each outperform their own buy-and-hold benchmarks in the verified four-and-a-half-year runs.
              </Callout>
              <Callout tone="warning" title="What Still Needs Work">
                The rotating one-asset-at-a-time selector improves over equal-weight buy-and-hold but is not yet better than the best standalone
                SOL profile on return or drawdown.
              </Callout>
              <Callout tone="danger" title="What Not To Infer">
                Do not infer that the ETH or SOL profile transfers unchanged to other assets. Cross-asset checks showed material degradation.
              </Callout>
            </div>
          </Section>

          <Section id="mechanics" title="Strategy Mechanics" subtitle="The core signal is a layered transformation from trend votes to target exposure.">
            <div className="three-col">
              <div className="info-card">
                <h3>1. Trend Stack</h3>
                <p>Closed-bar Supertrend votes are computed on weekly, daily, and 4h timeframes. BTC/SOL use the slower default stack; ETH uses faster daily/4h parameters.</p>
              </div>
              <div className="info-card">
                <h3>2. Score and Filter</h3>
                <p>Votes are averaged into a raw score, optionally filtered by RSI, clipped to [-1, 1], and zeroed inside the no-trade zone.</p>
              </div>
              <div className="info-card">
                <h3>3. Exposure Overlay</h3>
                <p>Volatility targeting scales exposure, applies long/short caps, and then applies bull floors, pullback floors, or short governors.</p>
              </div>
            </div>
            <div className="chart-spacing">
              <DataTable
                rows={data.profileConfigs as MetricRow[]}
                columns={[
                  { key: "asset", label: "Asset" },
                  { key: "supertrend", label: "Supertrend" },
                  { key: "rsi_filter", label: "RSI" },
                  { key: "target_vol_pct", label: "Target Vol", render: (row) => pct(row.target_vol_pct, 0) },
                  { key: "long_cap", label: "Long Cap", render: (row) => `${fmt(row.long_cap, 2)}x` },
                  { key: "short_cap", label: "Short Cap", render: (row) => `${fmt(row.short_cap, 2)}x` },
                  { key: "short_rule", label: "Short Rule" },
                ]}
              />
            </div>
          </Section>

          <Section id="traffic" title="Traffic-Light Engine" subtitle="The same traffic-light language now describes trend/exposure states rather than Kamino account states.">
            <TrafficStateCards states={data.trafficStates} />
          </Section>

          <Section id="risk" title="Risk Governors" subtitle="The governors are signal-level risk controls. They shape target exposure before exchange simulation.">
            <div className="two-col">
              <div className="info-card">
                <h3>Governor Stack</h3>
                <Checklist items={data.governors} />
              </div>
              <div className="info-card">
                <h3>Current Archetypes</h3>
                <p>
                  BTC is macro-trend, ETH is fast-transition, and SOL is high-beta near long-only. The next universe module should infer these
                  archetypes from liquidity, volatility, trend persistence, shortability, and convexity features.
                </p>
              </div>
            </div>
          </Section>

          <Section id="evidence" title="Evidence and Charts" subtitle="Current evidence is from verified profile CLI runs and the rotating long-only portfolio test.">
            <div className="chart-spacing">
              <ChartCard title="Profile Equity Curves" subtitle="Indexed to 100" legend={profileSeries.map((series) => ({ label: series.name, color: series.color }))}>
                <MultiLineChart series={profileSeries} />
              </ChartCard>
            </div>
            <div className="asset-diagnostic-grid">
              {data.assetDiagnostics.map((chart) => (
                <AssetDiagnosticChartCard chart={chart} key={chart.asset} />
              ))}
            </div>
            <div className="chart-spacing">
              <DataTable
                rows={profileRows}
                columns={[
                  { key: "asset", label: "Asset" },
                  { key: "archetype", label: "Archetype" },
                  { key: "total_return_pct", label: "Return", render: (row) => pct(row.total_return_pct, 2) },
                  { key: "buy_hold_pct", label: "B&H", render: (row) => pct(row.buy_hold_pct, 2) },
                  { key: "max_drawdown_pct", label: "Max DD", render: (row) => pct(row.max_drawdown_pct, 2) },
                  { key: "sharpe", label: "Sharpe", render: (row) => fmt(row.sharpe, 2) },
                  { key: "sortino", label: "Sortino", render: (row) => fmt(row.sortino, 2) },
                  { key: "trades", label: "Trades", render: (row) => fmt(row.trades, 0) },
                ]}
              />
            </div>
          </Section>

          <Section id="regimes" title="Regime Playbook" subtitle="The signal should be evaluated by archetype and regime, not only full-period return.">
            <div className="three-col">
              <div className="info-card">
                <h3>BTC-like</h3>
                <p>Use slower macro trend confirmation, lower target volatility, weekly-bull short blocks, and modest pullback exposure.</p>
              </div>
              <div className="info-card">
                <h3>ETH-like</h3>
                <p>Use faster daily/4h trend response, higher target volatility, and require weekly plus daily bearishness before shorting.</p>
              </div>
              <div className="info-card">
                <h3>SOL-like</h3>
                <p>Prefer long/flat behavior, higher target volatility, no current short cap, and small weekly-bull pullback participation.</p>
              </div>
            </div>
          </Section>

          <Section id="transfer" title="Portfolio Transfer Tests" subtitle="The first rotation test allows BTC, ETH, or SOL, but only one long asset at a time.">
            <DataTable
              rows={rotatingRows}
              columns={[
                { key: "name", label: "Selector" },
                { key: "rule", label: "Rule" },
                { key: "total_return_pct", label: "Return", render: (row) => pct(row.total_return_pct, 2) },
                { key: "max_drawdown_pct", label: "Max DD", render: (row) => pct(row.max_drawdown_pct, 2) },
                { key: "sharpe", label: "Sharpe", render: (row) => fmt(row.sharpe, 2) },
                { key: "sortino", label: "Sortino", render: (row) => fmt(row.sortino, 2) },
                { key: "rotations", label: "Rotations", render: (row) => fmt(row.rotations, 0) },
              ]}
            />
            <p className="footer-note">
              Interpretation: Selector A beat Selector B, but both had drawdowns above 52% and were not superior to the standalone SOL profile.
            </p>
          </Section>

          <Section id="usage" title="When To Use and When Not To Use" subtitle="This is a research checkpoint for signal and universe design, not a production deployment checklist.">
            <div className="two-col">
              <div className="info-card">
                <h3>Use When</h3>
                <Checklist items={data.useCases} />
              </div>
              <div className="info-card">
                <h3>Do Not Use When</h3>
                <Checklist items={data.nonUseCases} negative />
              </div>
            </div>
          </Section>

          <Section id="failures" title="Failure Modes" subtitle="Known ways this strategy family can give false confidence.">
            <div className="three-col">
              {data.failureModes.map((item, index) => (
                <div className="callout warning" key={item}>
                  <h3>Failure {index + 1}</h3>
                  <p>{item}</p>
                </div>
              ))}
            </div>
          </Section>

          <Section id="production" title="Production Readiness" subtitle="Universe design is the next mandatory gate before expanding beyond BTC, ETH, and SOL.">
            <div className="two-col">
              <div className="info-card">
                <h3>Required Controls</h3>
                <Checklist items={data.productionControls} />
              </div>
              <div className="info-card">
                <h3>Current Readiness Judgment</h3>
                <p>
                  Research-grade, not production-ready. The signal architecture is coherent, but the next phase must solve tradable universe
                  selection, historical survivorship bias, exchange constraints, and portfolio allocation before live deployment.
                </p>
              </div>
            </div>
          </Section>

          <Section id="audit" title="Audit Trail" subtitle="The displayed claims are tied to local reports and promoted profile code.">
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Artifact</th>
                  </tr>
                </thead>
                <tbody>
                  {data.artifactLinks.map((item) => (
                    <tr key={item}>
                      <td>
                        <code>{item}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="footer-note">Typed data module: apps/strategy-system-card/src/data/purePerpSignalSystemCardData.ts</p>
          </Section>
        </div>
      </main>
    </div>
  );
}

export function App() {
  const initialCard = useMemo(() => {
    const params = new URLSearchParams(window.location.search);
    return params.get("card") === "pure-perp" ? "pure-perp" : "kamino";
  }, []);
  const [selectedCard, setSelectedCard] = useState<"kamino" | "pure-perp">(initialCard);

  function selectCard(card: "kamino" | "pure-perp") {
    setSelectedCard(card);
    const url = new URL(window.location.href);
    url.searchParams.set("card", card);
    window.history.replaceState({}, "", url);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  if (selectedCard === "pure-perp") {
    return <PurePerpSystemCard selected={selectedCard} onSelect={selectCard} />;
  }
  return <KaminoSystemCard selected={selectedCard} onSelect={selectCard} />;
}
