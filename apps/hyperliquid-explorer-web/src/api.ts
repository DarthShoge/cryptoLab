import { useEffect, useState } from "react";
import type { components } from "./api.generated";
export type Run = components["schemas"]["Run"];
export type Detail = components["schemas"]["RunDetail"];
export type Scenario = components["schemas"]["Scenario"];
export type Metric = components["schemas"]["Metric"];
export type Analytics = components["schemas"]["Analytics"];
export type EquityPage = components["schemas"]["Page_EquityRow_"];
export type RecordPage = components["schemas"]["Page_Record_"];
export type Row = components["schemas"]["Record"];

export function scenarioParams(s: Scenario) {
  const params = new URLSearchParams({
    scenario_type: s.scenario_type,
    name: s.name,
  });
  if (s.latency_seconds !== null)
    params.set("latency_seconds", String(s.latency_seconds));
  return params;
}
export function runUrl(id: string, path = "") {
  return `/api/runs/${encodeURIComponent(id)}${path}`;
}
export function useResource<T>(url: string | null) {
  const [state, setState] = useState<{
    url: string | null;
    data?: T;
    error?: string;
  }>({ url: null });
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    if (!url) return;
    const controller = new AbortController();
    setState({ url });
    fetch(url, { signal: controller.signal, cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) {
          const body = (await response.json().catch(() => null)) as {
            detail?: unknown;
          } | null;
          throw new Error(
            typeof body?.detail === "string"
              ? body.detail
              : `Request failed (${response.status})`,
          );
        }
        return response.json() as Promise<T>;
      })
      .then((data) => {
        if (!controller.signal.aborted) setState({ url, data });
      })
      .catch((error) => {
        if (!controller.signal.aborted)
          setState({
            url,
            error:
              error instanceof Error ? error.message : "Unable to load data",
          });
      });
    return () => controller.abort();
  }, [url, attempt]);
  return {
    ...(state.url === url ? state : {}),
    loading:
      Boolean(url) && (state.url !== url || (!state.data && !state.error)),
    retry: () => setAttempt((n) => n + 1),
  };
}
