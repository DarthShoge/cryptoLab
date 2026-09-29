import { expect, it } from "vitest";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ScenarioSelect } from "./Overview";
import { experimentLabel, type Experiment } from "./labApi";
import type { Scenario } from "./api";

it("describes the saved hypothesis and distinguishes repeated runs", () => {
  const experiment = {
    name: "My hypothesis",
    id: "abcdef1234567890",
    config: {
      schema_version: "hyperliquid_copy_lab_v1",
      coins: ["ETH", "SOL"],
      selection: "n",
      top_n: 2,
      lookback_days: 30,
      scope: "per_asset",
      reselection: "weekly",
      aggregation: "direction_equal",
      start: "2026-01-01",
      end: "2026-02-01",
    },
  } as Experiment;
  const result = experimentLabel(experiment);
  for (const part of [
    "My hypothesis",
    "ETH + SOL",
    "top 2",
    "30d",
    "weekly",
    "equal-weight direction",
    "2026-01-01",
    "abcdef1234",
  ])
    expect(result).toContain(part);
  expect(experimentLabel({ ...experiment, id: "9876543210abcdef" })).not.toBe(
    result,
  );
});

it("uses the descriptive label without changing scenario identity or benchmark labels", () => {
  const strategy = {
    scenario_type: "strategy",
    name: "direction_equal",
    latency_seconds: 5,
  } as Scenario;
  const cash = {
    scenario_type: "control",
    name: "cash",
    latency_seconds: null,
  } as Scenario;
  const markup = renderToStaticMarkup(
    createElement(ScenarioSelect, {
      scenarios: [strategy, cash],
      selected: strategy,
      onChange: () => {},
      strategyLabel: "My descriptive hypothesis",
    }),
  );
  expect(markup).toContain("My descriptive hypothesis");
  expect(markup).toContain('value="strategy:direction_equal"');
  expect(markup).toContain("Cash · benchmark");
  const legacy = renderToStaticMarkup(
    createElement(ScenarioSelect, {
      scenarios: [strategy],
      selected: strategy,
      onChange: () => {},
    }),
  );
  expect(legacy).toContain("Direction equal");
});
