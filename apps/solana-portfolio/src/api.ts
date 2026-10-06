import { useCallback, useEffect, useRef, useState } from "react";
import type { Asset, ChartData, Interval, Mode, Portfolio } from "./types";

export async function request<T>(url: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(url, options);
  if (!response.headers.get("content-type")?.includes("application/json")) throw new Error("The portfolio API is unavailable. Start the app with npm run dev:portfolio.");
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "Unable to load data.");
  return data as T;
}

export function usePortfolio() {
  const [mode, setMode] = useState<Mode>("demo");
  const [wallet, setWallet] = useState("");
  const [data, setData] = useState<Portfolio | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [revision, setRevision] = useState(0);
  const controller = useRef<AbortController | null>(null);
  useEffect(() => {
    const abort = new AbortController();
    request<{ defaultWallet: string }>("/api/config", { signal: abort.signal }).then(config => setWallet(current => current || config.defaultWallet)).catch(() => {});
    return () => abort.abort();
  }, []);
  useEffect(() => {
    if (mode === "imported") return;
    const abort = new AbortController(); controller.current = abort;
    setLoading(true); setError(""); setData(null);
    request<Portfolio>(`/api/portfolio?${new URLSearchParams({ mode, ...(revision ? { refresh: "1" } : {}), ...(wallet ? { wallet } : {}) })}`, { signal: abort.signal })
      .then(setData).catch(reason => { if (!abort.signal.aborted) setError(reason.message); })
      .finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => abort.abort();
  }, [mode, mode === "live" ? wallet : "", revision]);
  useEffect(() => {
    if (mode !== "live" || !data?.wallet) return;
    const account = data.wallet;
    const abort = new AbortController();
    let pending = false;
    const timer = window.setInterval(async () => {
      if (pending) return;
      pending = true;
      try {
        const history = await request<Pick<Portfolio, "trades" | "historyStatus">>(`/api/history?${new URLSearchParams({ wallet: account })}`, { signal: abort.signal });
        if (!abort.signal.aborted) setData(current => current?.wallet === account && current.mode === "live" ? { ...current, ...history } : current);
      } catch { /* Existing coverage remains visible; provider errors appear in history status. */ }
      finally { pending = false; }
    }, 5000);
    return () => { window.clearInterval(timer); abort.abort(); };
  }, [mode, data?.wallet]);
  const importHistory = useCallback(async (file: File) => {
    if (file.size > 2_000_000) throw new Error("Upload a JSON history smaller than 2 MB.");
    const body = await file.text();
    const imported = await request<Portfolio>("/api/import", { method: "POST", headers: { "Content-Type": "application/json" }, body });
    controller.current?.abort(); setMode("imported"); setData(imported); setLoading(false); setError("");
  }, []);
  return { mode, setMode, wallet, setWallet, data, loading, error, refresh: () => setRevision(n => n + 1), importHistory };
}

export function useChart(mode: Mode, asset: Asset, interval: Interval, days: number, period: number, multiplier: number, end?: number, start?: number, combined = false) {
  const [data, setData] = useState<ChartData | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const abort = new AbortController();
    setLoading(true); setError(""); setData(null);
    request<ChartData>(`/api/chart?${new URLSearchParams({ mode, asset, interval, days: String(days), period: String(period), multiplier: String(multiplier), ...(combined ? {combined: "1"} : {}), ...(mode === "imported" && end ? { end: String(end) } : {}), ...(days === 0 && start ? { start: String(start) } : {}) })}`, { signal: abort.signal })
      .then(setData).catch(reason => { if (!abort.signal.aborted) setError(reason.message); })
      .finally(() => { if (!abort.signal.aborted) setLoading(false); });
    return () => abort.abort();
  }, [mode, asset, interval, days, period, multiplier, end, start, combined]);
  return { data, error, loading };
}
