import { useEffect, useState } from "react";
import type { components } from "./api.generated";
export type Config = Required<components["schemas"]["LabConfig"]>;
export type Experiment = components["schemas"]["Experiment"];
export type Dataset = components["schemas"]["Dataset"];
export type Bootstrap = components["schemas"]["Bootstrap"];
export type Submission = components["schemas"]["Submission"];
export type UniverseRow = components["schemas"]["UniverseRow"];
export type UniversePage = components["schemas"]["Page_UniverseRow_"];
export type Comparison = components["schemas"]["Comparison"];

export async function mutate<T>(
  path: string,
  token: string,
  body: unknown = {},
  method = "POST",
): Promise<T> {
  const response = await fetch(`/api/lab${path}`, {
    method,
    headers: { "Content-Type": "application/json", "X-Lab-Token": token },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    if (response.status === 403)
      throw new Error(
        "The local session is unavailable or expired. Refresh the page before submitting.",
      );
    const detail =
      typeof data.detail === "string"
        ? data.detail
        : Array.isArray(data.detail)
          ? data.detail.map((d: { msg: string }) => d.msg).join("; ")
          : "Request failed";
    throw new Error(detail);
  }
  return data as T;
}

export function useExperiment(id: string | null) {
  const [state, setState] = useState<{
    id: string | null;
    data?: Experiment;
    error?: string;
  }>({ id: null });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout>;
    setState({ id });
    const poll = async () => {
      try {
        const response = await fetch(`/api/lab/experiments/${id}`, {
          signal: controller.signal,
          cache: "no-store",
        });
        if (!response.ok) throw new Error("Unable to load saved backtest");
        const data = (await response.json()) as Experiment;
        if (controller.signal.aborted) return;
        setState({ id, data });
        if (data.status === "queued" || data.status === "running")
          timer = setTimeout(poll, 500);
      } catch (error) {
        if (!controller.signal.aborted) setState({ id, error: String(error) });
      }
    };
    void poll();
    return () => {
      controller.abort();
      clearTimeout(timer);
    };
  }, [id, attempt]);
  return {
    ...(state.id === id ? state : {}),
    loading: !!id && (state.id !== id || (!state.data && !state.error)),
    retry: () => setAttempt((n) => n + 1),
  };
}

export function strategySummary(config: Config) {
  const cohort =
    config.selection === "n"
      ? `top ${config.top_n}`
      : `top ${(config.top_fraction ?? 0) * 100}%`;
  return `${config.coins.join(" + ")} · ${config.scope.replace("_", " ")} · ${cohort} eligible · ${config.lookback_days}d · ${config.reselection} · ${config.aggregation.replace(/_/g, " ")} · BTC perp benchmark`;
}

export function experimentLabel(experiment: Experiment) {
  const config = experiment.config as Config;
  const methods = {
    direction_equal: "equal-weight direction copying",
    direction_score_weighted: "score-weighted direction copying",
    conviction_trimmed: "trimmed-conviction copying",
  };
  const summary = strategySummary(config).replace(
    config.aggregation.replace(/_/g, " "),
    methods[config.aggregation],
  );
  return `${experiment.name} — ${summary} · ${config.start} → ${config.end} · #${experiment.id.slice(0, 10)}`;
}
