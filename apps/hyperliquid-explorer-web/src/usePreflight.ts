import { useEffect, useRef, useState } from "react";
import type { components } from "./api.generated";
import type { WireConfig } from "./marketConfig";

export type Preflight = components["schemas"]["Preflight"];

export function usePreflight(
  datasetId: string,
  config: WireConfig,
  token: string,
) {
  const key = JSON.stringify({ dataset_id: datasetId, config });
  const generation = useRef(0);
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<{
    key: string;
    token: string;
    data?: Preflight;
    error?: string;
  }>();
  useEffect(() => {
    const current = ++generation.current;
    const controller = new AbortController();
    setState(undefined);
    if (!datasetId) return;
    const timer = setTimeout(async () => {
      try {
        const response = await fetch("/api/lab/preflight", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Lab-Token": token },
          body: key,
          signal: controller.signal,
        });
        const body = await response.json();
        if (!response.ok) {
          if (response.status === 403)
            throw new Error(
              "The local session is unavailable or expired. Refresh the page before submitting.",
            );
          throw new Error(
            typeof body.detail === "string"
              ? body.detail
              : Array.isArray(body.detail)
                ? body.detail
                    .map((issue: { msg: string }) => issue.msg)
                    .join("; ")
                : "Unable to validate this draft.",
          );
        }
        if (current === generation.current && !controller.signal.aborted)
          setState({ key, token, data: body });
      } catch (error) {
        if (current === generation.current && !controller.signal.aborted)
          setState({
            key,
            token,
            error:
              error instanceof Error
                ? error.message
                : "Unable to check this draft.",
          });
      }
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [key, datasetId, token, attempt]);
  const current =
    state?.key === key && state.token === token ? state : undefined;
  return {
    data: current?.data,
    error: current?.error,
    ready: !!datasetId && current?.data?.ready === true,
    checking: !!datasetId && !current,
    missingDataset: !datasetId,
    retry: () => {
      setState(undefined);
      setAttempt((value) => value + 1);
    },
  };
}
