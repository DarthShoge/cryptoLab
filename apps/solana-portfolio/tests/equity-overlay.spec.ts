import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import ts from "typescript";

const require = createRequire(import.meta.url);
const app = resolve(dirname(fileURLToPath(import.meta.url)), "..");

function loadModule(filename: string): any {
  const output = ts.transpileModule(readFileSync(filename, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText;
  const module = { exports: {} };
  const imports = (name: string) => name === "../chartTime"
    ? loadModule(resolve(app, "src/chartTime.ts")) : require(name);
  new Function("require", "module", "exports", output)(imports, module, module.exports);
  return module.exports;
}

function render(history: any[], extra: Record<string, unknown> = {}) {
  const { EquityOverlay } = loadModule(resolve(app, "src/components/EquityOverlay.tsx"));
  return renderToStaticMarkup(React.createElement(EquityOverlay, {
    candles: [
      {time: 0, endTime: 86400, open: 100, high: 200, low: 50, close: 150, volume: 1},
      {time: 86400, endTime: 172800, open: 100, high: 200, low: 50, close: 150, volume: 1},
    ],
    history, left: 20, right: 30, width: 350, top: 10, bottom: 110,
    interval: "1d", label: "Known portfolio equity", ...extra,
  }));
}

test("equity overlay leaves absent equity unavailable without infinite SVG coordinates", () => {
  const output = render([{time: 0, equity: null}, {time: 86400, equity: NaN}]);
  expect(output).toContain("Equity USD");
  expect(output).toContain('data-no-equity="true"');
  expect(output).not.toContain("chart-equity-line");
  expect(output).not.toMatch(/NaN|Infinity/);
});

test("equity overlay plots an isolated constant observation on an independent finite scale", () => {
  const output = render([{time: 43200, equity: 1500}]);
  expect(output).toContain("Known portfolio equity");
  expect(output).toContain('class="chart-equity-line"');
  expect(output).toContain('cx="95"');
  expect(output).toContain('cy="60"');
  expect(output).not.toMatch(/NaN|Infinity/);
});

test("equity overlay does not bridge invalid values or changes in valuation coverage", () => {
  const output = render([
    {time: 1000, equity: 10, valuationComplete: true, unpricedCount: 0},
    {time: 2000, equity: null, valuationComplete: true, unpricedCount: 0},
    {time: 3000, equity: 12, valuationComplete: true, unpricedCount: 0},
    {time: 4000, equity: 13, valuationComplete: false, unpricedCount: 1},
  ]);
  expect(output.match(/class="chart-equity-line"/g)).toHaveLength(3);
  expect(output).not.toContain("<polyline");
});

test("equity overlay respects calendar candle ends and keeps daily missing buckets disconnected", () => {
  const january = Date.UTC(2025, 0, 1) / 1000;
  const february = Date.UTC(2025, 1, 1) / 1000;
  const output = render([
    {time: january, equity: 10},
    {time: january + 15.5 * 86400, equity: 12},
  ], {
    candles: [{time: january, endTime: february, open: 1, high: 2, low: 1, close: 2, volume: 1}],
    interval: "1M", dailyBuckets: true, label: "Kamino net equity",
  });
  expect(output).toContain('cx="170"');
  expect(output.match(/class="chart-equity-line"/g)).toHaveLength(2);
  expect(output).toContain("daily bucket");
});
