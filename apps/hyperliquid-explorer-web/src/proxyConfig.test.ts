import { expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { editorConfig, universeOf, wireConfig, type ConfigProxy, type ConfigProxyScheduled } from "./marketConfig";
import { strategySummary, type Bootstrap, type Dataset } from "./labApi";
import { StrategyBuilder } from "./StrategyBuilder";

const proxy = {
  schema_version: "hyperliquid_copy_lab_proxy_v1",
  market_universe: { mode: "explicit", general: false, classes: ["equity"], instrument_ids: ["xyz:TSLA"], allocation: "equal", weights: null, reselection: "daily" },
  trader: { scope: "per_asset", selection: "n", top_n: 5, lookback_days: 30, reselection: "daily", metric_weights: {}, metric_directions: {} },
  follower: { aggregation: "direction_equal", update_minutes: 60, initial_equity: 10000 },
  proxy: { slippage_bps: 25, max_mark_age_seconds: 345600, max_wait_seconds: 345600 },
  start: "2026-08-03", end: "2026-08-07", benchmark: "btc_perp_buy_hold", split: "development",
} as ConfigProxy;

it("preserves proxy identity, settings and markets through the editor", () => {
  const editor = editorConfig(proxy);
  expect(universeOf(proxy)).toEqual(proxy.market_universe);
  const saved = wireConfig(editor, universeOf(proxy), proxy.proxy);
  expect(saved.schema_version).toBe(proxy.schema_version);
  expect(saved).toHaveProperty("proxy", proxy.proxy);
  expect(saved.follower.update_minutes).toBe(60);
  expect(strategySummary(proxy)).toContain("xyz:TSLA");
  expect(strategySummary(proxy)).toContain("hourly proxy");
});

it("shows hourly execution and mapping assumptions for a cloned proxy draft", () => {
  const dataset: Dataset = { id: "proxy", name: "Hourly data", available: true, synthetic: false, liquidity_available: true, rows: 100, pricing_mode: "hourly_proxy", coins: ["BTC", "xyz:TSLA"], coverage_note: "Observed native activity", proxy_mappings: [{instrument_id: "xyz:TSLA", ticker: "TSLA", provider: "yahoo", unit: "share", calendar: "XNYS"}] };
  const markup = renderToStaticMarkup(createElement(StrategyBuilder, {
    bootstrap: { defaults: proxy, metrics: [], token: "test" } as unknown as Bootstrap,
    datasets: [dataset], initial: {name: "Hourly clone", dataset_id: "proxy", config: proxy}, onRun: () => {},
  }));
  expect(markup).toContain("APPROXIMATE HOURLY PROXY");
  expect(markup).toContain("Hourly (60 minutes)");
  expect(markup).toContain('aria-label="Proxy slippage (basis points)"');
  expect(markup).toContain('value="25"');
  expect(markup).toContain("XNYS");
  expect(markup).toContain("not historical listing dates");
});

it("preserves weekly target cadence separately from hourly valuation", () => {
  const scheduled = {...proxy, schema_version:"hyperliquid_copy_lab_proxy_v2", rebalance:"weekly",
    trader:{...proxy.trader,reselection:"weekly"},market_universe:{...proxy.market_universe,reselection:"weekly"}} as ConfigProxyScheduled;
  const saved=wireConfig(editorConfig(scheduled),universeOf(scheduled),scheduled.proxy,scheduled.rebalance);
  expect(saved.schema_version).toBe("hyperliquid_copy_lab_proxy_v2");
  expect(saved).toHaveProperty("rebalance","weekly");
  expect(saved.follower.update_minutes).toBe(60);
  expect(strategySummary(scheduled)).toContain("weekly trader/portfolio");
});
