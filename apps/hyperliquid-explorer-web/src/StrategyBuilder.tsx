import { useEffect, useRef, useState } from "react";
import {
  type Bootstrap,
  type Config,
  type Dataset,
  type Experiment,
  type Submission,
  mutate,
  strategySummary,
  useExperiment,
} from "./labApi";
import { label } from "./format";
import { Status } from "./Status";
import { RankingTable } from "./Universe";
import { usePreflight } from "./usePreflight";
import { RunPreflight } from "./RunPreflight";

type Props = {
  bootstrap: Bootstrap;
  datasets: Dataset[];
  initial?: Submission;
  onRun: (e: Experiment) => void;
};
export function StrategyBuilder({
  bootstrap,
  datasets,
  initial,
  onRun,
}: Props) {
  const [config, setConfig] = useState<Config>(
    (initial?.config ?? bootstrap.defaults) as Config,
  );
  const [name, setName] = useState(
    initial?.name ?? "My trader-copy hypothesis",
  );
  const [datasetId, setDatasetId] = useState(
    initial?.dataset_id ?? datasets.find((d) => d.available)?.id ?? "",
  );
  const [error, setError] = useState(""),
    [busy, setBusy] = useState(false),
    [previewId, setPreviewId] = useState<string | null>(null);
  const [previewDate, setPreviewDate] = useState(config.start),
    [previewScope, setPreviewScope] = useState(config.coins[0]);
  const preview = useExperiment(previewId);
  const dataset = datasets.find((d) => d.id === datasetId);
  const preflight = usePreflight(
    dataset?.available ? datasetId : "",
    config,
    bootstrap.token,
  );
  const submitting = useRef(false);
  const errorRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (error) errorRef.current?.focus();
  }, [error]);
  const set = <K extends keyof Config>(key: K, value: Config[K]) =>
    setConfig((c) => ({ ...c, [key]: value }));
  const numeric = (key: keyof Config, title: string, step = "any") => (
    <label key={key}>
      {title}
      <input
        aria-label={title}
        type="number"
        step={step}
        value={typeof config[key] === "number" ? String(config[key]) : ""}
        onChange={(e) => set(key, Number(e.target.value) as never)}
      />
    </label>
  );
  const choose = (key: keyof Config, title: string, options: string[]) => (
    <label>
      {title}
      <select
        aria-label={title}
        value={String(config[key])}
        onChange={(e) => set(key, e.target.value as never)}
      >
        {options.map((v) => (
          <option key={v} value={v}>
            {label(v)}
          </option>
        ))}
      </select>
    </label>
  );
  const run = async (isPreview = false) => {
    if (submitting.current || !preflight.ready) return;
    submitting.current = true;
    setBusy(true);
    setError("");
    try {
      const body = {
        name,
        dataset_id: datasetId,
        config,
        parent_id: initial?.parent_id,
      };
      const result = await mutate<Experiment>(
        isPreview ? "/previews" : "/experiments",
        bootstrap.token,
        isPreview
          ? {
              ...body,
              decision_date: previewDate,
              scope: config.scope === "pooled" ? null : previewScope,
            }
          : body,
      );
      if (isPreview) setPreviewId(result.id);
      else onRun(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  };
  const preset = () => {
    if (dataset?.default_config) {
      const c = dataset.default_config as Config;
      setConfig(c);
      setPreviewDate(c.start);
      setPreviewScope(c.coins[0]);
      setError("");
      setPreviewId(null);
    }
  };
  return (
    <>
      <div className="hypothesis-strip">
        <span className="eyebrow">EDITABLE STRATEGY DRAFT</span>
        <p>{strategySummary(config)}</p>
      </div>
      <div className="builder-layout">
        <div>
          <section className="panel">
            <h2>Trader universe & selection</h2>
            <p className="muted">
              Define whose positions to copy. Selection is recomputed using only
              data available before each decision.
            </p>
            <label>
              Hypothesis name
              <input
                aria-label="Hypothesis name"
                maxLength={120}
                value={name}
                onChange={(e) => setName(e.target.value)}
              />
            </label>
            <fieldset>
              <legend>Copied markets</legend>
              <div className="check-row">
                {["BTC", "ETH", "SOL"].map((coin) => (
                  <label key={coin}>
                    <input
                      type="checkbox"
                      checked={config.coins.includes(coin)}
                      onChange={(e) => {
                        const coins = e.target.checked
                          ? [...config.coins, coin].sort()
                          : config.coins.filter((c) => c !== coin);
                        setConfig((c) => ({
                          ...c,
                          coins,
                          asset_weights: Object.fromEntries(
                            coins.map((c) => [c, 1 / coins.length]),
                          ),
                        }));
                        if (!e.target.checked && previewScope === coin)
                          setPreviewScope(coins[0] ?? "");
                      }}
                    />
                    {coin}
                  </label>
                ))}
              </div>
              <small>
                BTC benchmark data is independent of these copied markets.
              </small>
            </fieldset>
            <div className="form-grid">
              {choose("scope", "Ranking scope", ["per_asset", "pooled"])}
              {numeric("lookback_days", "Trailing lookback (days)", "1")}
              <label>
                Selection rule
                <select
                  aria-label="Selection rule"
                  value={config.selection}
                  onChange={(e) =>
                    setConfig((c) => ({
                      ...c,
                      selection: e.target.value as Config["selection"],
                      top_n: e.target.value === "n" ? 5 : null,
                      top_fraction: e.target.value === "fraction" ? 0.05 : null,
                    }))
                  }
                >
                  <option value="fraction">
                    Top percentage of eligible wallets
                  </option>
                  <option value="n">Top N eligible wallets</option>
                </select>
              </label>
              {config.selection === "n" ? (
                numeric("top_n", "Top N traders", "1")
              ) : (
                <label>
                  Top eligible traders (%)
                  <input
                    aria-label="Top eligible traders (%)"
                    type="number"
                    step="any"
                    value={(config.top_fraction ?? 0) * 100}
                    onChange={(e) =>
                      set("top_fraction", Number(e.target.value) / 100)
                    }
                  />
                </label>
              )}
              {numeric("min_cohort", "Minimum cohort size", "1")}
              {numeric("max_cohort", "Maximum cohort size", "1")}
              {choose("reselection", "Universe reselection", [
                "daily",
                "weekly",
                "monthly",
              ])}
            </div>
            <details>
              <summary>Eligibility thresholds</summary>
              <div className="form-grid">
                {numeric("min_active_days", "Minimum active days", "1")}
                {numeric("min_episodes", "Minimum completed episodes", "1")}
                {numeric("min_notional", "Minimum closing notional (USD)")}
                {numeric("min_volume", "Minimum gross traded volume (USD)")}
                {numeric("min_minutes", "Minimum median holding minutes")}
              </div>
            </details>
          </section>
          <section className="panel">
            <h2>What makes a top trader?</h2>
            <p className="muted">
              Weighted percentile scores within each eligible cohort. Set zero
              to exclude a metric; the backend normalises positive weights.
            </p>
            <div className="table-scroll">
              <table>
                <thead>
                  <tr>
                    <th>Ranking metric</th>
                    <th>Weight</th>
                    <th>Preference</th>
                  </tr>
                </thead>
                <tbody>
                  {bootstrap.metrics.map((metric) => (
                    <tr key={metric}>
                      <th>{label(metric)}</th>
                      <td>
                        <input
                          aria-label={`${label(metric)} weight`}
                          type="number"
                          min="0"
                          step="any"
                          value={config.metric_weights[metric] ?? 0}
                          onChange={(e) =>
                            set("metric_weights", {
                              ...config.metric_weights,
                              [metric]: Number(e.target.value),
                            })
                          }
                        />
                      </td>
                      <td>
                        <select
                          aria-label={`${label(metric)} preference`}
                          value={config.metric_directions[metric] ?? "desc"}
                          onChange={(e) =>
                            set("metric_directions", {
                              ...config.metric_directions,
                              [metric]: e.target.value as "asc" | "desc",
                            })
                          }
                        >
                          <option value="desc">Higher is better</option>
                          <option value="asc">Lower is better</option>
                        </select>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="notice">
              Wallet return / ROI: unavailable. PnL efficiency is realised PnL
              per closing notional, not return on account equity. Volume
              measures activity, not skill.
            </p>
          </section>
          <section className="panel">
            <h2>Copying & portfolio construction</h2>
            <div className="form-grid">
              {choose("aggregation", "Trader weighting", [
                "direction_equal",
                "direction_score_weighted",
                "conviction_trimmed",
              ])}
              {numeric(
                "update_minutes",
                "Target update cadence (minutes)",
                "1",
              )}
              {numeric("gross_cap", "Gross exposure budget (multiple)")}
              {numeric("asset_cap", "Per-asset exposure cap (multiple)")}
              {config.coins.map((coin) => (
                <label key={coin}>
                  {coin} asset budget
                  <input
                    aria-label={`${coin} asset budget`}
                    type="number"
                    step="any"
                    value={config.asset_weights[coin]}
                    onChange={(e) =>
                      set("asset_weights", {
                        ...config.asset_weights,
                        [coin]: Number(e.target.value),
                      })
                    }
                  />
                </label>
              ))}
            </div>
            <small>
              Asset budgets must sum to one. Equal trader votes and equal asset
              allocation are different settings.
            </small>
            <details>
              <summary>Coverage, normalisation & execution</summary>
              <div className="form-grid">
                {numeric("min_known", "Minimum known trader positions", "1")}
                {numeric("min_known_weight", "Minimum known weight (fraction)")}
                {numeric("latency_seconds", "Execution delay (seconds)", "1")}
                {numeric("fee_bps", "Follower fee (basis points)")}
                {numeric("deadband", "Rebalance deadband (fraction)")}
                {numeric("min_trade_usd", "Minimum trade (USD)")}
                {config.aggregation === "conviction_trimmed" && (
                  <>
                    {numeric(
                      "scale_lookback_days",
                      "Conviction scale lookback (days)",
                      "1",
                    )}
                    {numeric(
                      "scale_quantile",
                      "Trailing-notional scale quantile",
                    )}
                    {numeric("trim", "Trim per tail (fraction)")}
                  </>
                )}
              </div>
            </details>
          </section>
        </div>
        <aside className="builder-side">
          <section className="panel">
            <h2>Backtest data & assumptions</h2>
            <label>
              Local dataset
              <select
                aria-label="Local dataset"
                value={datasetId}
                onChange={(e) => {
                  setDatasetId(e.target.value);
                  setPreviewId(null);
                }}
              >
                <option value="">Select dataset</option>
                {datasets.map((d) => (
                  <option value={d.id} key={d.id} disabled={!d.available}>
                    {d.name}
                  </option>
                ))}
              </select>
            </label>
            {!datasets.length && (
              <p className="notice">No registered local datasets</p>
            )}
            {dataset && (
              <>
                <p className="mode-badge demo">
                  {dataset.synthetic
                    ? "SYNTHETIC DATASET"
                    : "UNQUALIFIED LOCAL DATA"}
                </p>
                <p className="muted">
                  {dataset.coverage_start} → {dataset.coverage_end}
                  <br />
                  {dataset.rows?.toLocaleString()} input rows ·{" "}
                  {dataset.coins?.join(" / ")}
                </p>
                <p className="muted">{dataset.coverage_note}</p>
                {dataset.synthetic && dataset.default_config && (
                  <button onClick={preset}>Load synthetic preset</button>
                )}
              </>
            )}
            <p className="muted">
              The synthetic preset deliberately relaxes eligibility and uses a
              one-day lookback. It is not a quarterly research result.
            </p>
            <div className="form-grid">
              <label>
                Backtest start (UTC)
                <input
                  aria-label="Backtest start (UTC)"
                  type="date"
                  value={config.start}
                  onChange={(e) => {
                    set("start", e.target.value);
                    setPreviewDate(e.target.value);
                  }}
                />
              </label>
              <label>
                Backtest end, exclusive (UTC)
                <input
                  aria-label="Backtest end, exclusive (UTC)"
                  type="date"
                  value={config.end}
                  onChange={(e) => set("end", e.target.value)}
                />
              </label>
              {numeric("initial_equity", "Starting capital (USD)")}
            </div>
            <p className="muted">
              Benchmark: BTC perpetual buy-and-hold, same period and initial
              capital. Development split only. Full warmup is required.
            </p>
            <RunPreflight state={preflight} />
            {error && (
              <div
                className="notice error"
                role="alert"
                tabIndex={-1}
                ref={errorRef}
              >
                {error}
              </div>
            )}
            <button
              className="primary full-width"
              disabled={busy || !preflight.ready || !config.coins.length}
              onClick={() => void run()}
            >
              {busy ? "Saving…" : "Run and save backtest"}
            </button>
            <small>
              Inputs are frozen on submission. No orders or paid data requests.
            </small>
          </section>
          <section className="panel">
            <h2>Inspect selection before running</h2>
            <label>
              Preview decision date
              <input
                aria-label="Preview decision date"
                type="date"
                value={previewDate}
                onChange={(e) => setPreviewDate(e.target.value)}
              />
            </label>
            {config.scope === "per_asset" && (
              <label>
                Preview asset
                <select
                  aria-label="Preview asset"
                  value={previewScope}
                  onChange={(e) => setPreviewScope(e.target.value)}
                >
                  {config.coins.map((c) => (
                    <option key={c}>{c}</option>
                  ))}
                </select>
              </label>
            )}
            <button
              className="full-width"
              disabled={busy || !preflight.ready}
              onClick={() => void run(true)}
            >
              Preview trader universe
            </button>
          </section>
        </aside>
      </div>
      {previewId && (
        <section className="panel">
          <h2>Historical cohort preview</h2>
          <Status {...preview} />
          {preview.data && (
            <p>
              <span>
                {preview.data.status === "completed"
                  ? "Preview complete"
                  : preview.data.status}
              </span>
              {
                " — frozen historical preview; subsequent draft edits do not change it. "
              }
              {strategySummary(preview.data.config as Config)}
            </p>
          )}
          {preview.data?.error && (
            <p className="notice error">{preview.data.error}</p>
          )}
          {preview.data?.status === "completed" && (
            <RankingTable id={previewId} />
          )}
        </section>
      )}
    </>
  );
}
