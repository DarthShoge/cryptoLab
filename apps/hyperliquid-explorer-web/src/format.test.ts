import { describe, it, expect } from "vitest";
import { formatValue, scenarioKey, compareNullable } from "./format";
describe("report presentation", () => {
  it("distinguishes unavailable from zero", () => {
    expect(formatValue(null, "percent")).toBe("N/A");
    expect(formatValue(0, "percent")).toBe("0.00%");
  });
  it("shows ratios and durations in their own units", () => {
    expect(formatValue(1.2, "ratio")).toBe("1.20");
    expect(formatValue(90, "minutes")).toBe("1h 30m");
  });
  it("keeps cash null distinct from zero latency", () => {
    expect(
      scenarioKey({
        scenario_type: "control",
        name: "cash",
        latency_seconds: null,
      }),
    ).not.toBe(
      scenarioKey({
        scenario_type: "control",
        name: "cash",
        latency_seconds: 0,
      }),
    );
  });
  it("sorts unavailable last in either direction", () => {
    expect(compareNullable(null, 2, true)).toBeGreaterThan(0);
    expect(compareNullable(null, 2, false)).toBeGreaterThan(0);
  });
});
