import type { Scenario } from "./api";
export function formatValue(
  value: number | null | undefined,
  unit: string,
): string {
  if (value == null || !Number.isFinite(value)) return "N/A";
  if (unit === "usd")
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 2,
    }).format(value);
  if (unit === "percent") return `${(value * 100).toFixed(2)}%`;
  if (unit === "minutes")
    return value >= 1440
      ? `${Math.floor(value / 1440)}d ${Math.floor((value % 1440) / 60)}h`
      : value >= 60
        ? `${Math.floor(value / 60)}h ${Math.floor(value % 60)}m`
        : `${value.toFixed(0)}m`;
  return unit === "count" ? value.toLocaleString("en-US") : value.toFixed(2);
}
export function scenarioKey(
  s: Pick<Scenario, "scenario_type" | "name" | "latency_seconds">,
) {
  return `${s.scenario_type}:${s.name}:${s.latency_seconds === null ? "null" : s.latency_seconds}`;
}
export function compareNullable(
  a: number | null | undefined,
  b: number | null | undefined,
  desc: boolean,
) {
  if (a == null) return b == null ? 0 : 1;
  if (b == null) return -1;
  return desc ? b - a : a - b;
}
export function label(key: string) {
  return (
    (
      {
        sharpe: "Sharpe",
        sortino: "Sortino",
        calmar: "Calmar",
        max_drawdown: "Max drawdown",
        final_equity: "Final equity",
        total_return: "Total return",
        time_net_flat: "Time net-neutral",
        annualized_return: "Annualized return (CAGR)",
      } as Record<string, string>
    )[key] ?? key.replace(/_/g, " ").replace(/^./, (s) => s.toUpperCase())
  );
}
export function utc(value: string | null | undefined) {
  return value
    ? new Date(value).toISOString().replace("T", " ").replace(".000Z", " UTC")
    : "Unavailable";
}
