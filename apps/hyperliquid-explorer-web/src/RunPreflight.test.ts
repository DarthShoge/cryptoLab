import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { RunPreflight } from "./RunPreflight";

describe("RunPreflight capacity", () => {
  it("labels the bound and preserves its limitations", () => {
    const html = renderToStaticMarkup(createElement(RunPreflight, {
      state: {
        ready: true,
        checking: false,
        missingDataset: false,
        error: undefined,
        retry: () => {},
        data: {
          ready: true,
          issues: [],
          config_hash: "test",
          required_start: "2025-06-02",
          required_end: "2026-09-01",
          estimates: { ranking_rows: 123456789 },
          estimate_notes: ["Row capacity does not establish total disk capacity."],
        },
      },
    }));
    expect(html).toContain("Ranking rows (upper bound):");
    expect(html).toContain("123,456,789");
    expect(html).toContain("Row capacity does not establish total disk capacity.");
  });
});
