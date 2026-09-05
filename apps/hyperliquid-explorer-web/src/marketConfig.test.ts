import { describe, expect, it } from "vitest";
import {
  changeClass,
  changeMode,
  generalUniverse,
  marketSummary,
  type MarketUniverse,
} from "./marketConfig";

const explicit: MarketUniverse = {
  mode: "explicit",
  general: false,
  classes: ["crypto"],
  instrument_ids: ["ETH"],
  allocation: "custom",
  weights: { ETH: 1 },
  reselection: "weekly",
};
describe("market draft transitions", () => {
  it("General drops selected classes, IDs and budgets", () => {
    const next = generalUniverse(explicit);
    expect(next.general).toBe(true);
    expect(next.classes).toEqual([]);
    expect(next).not.toHaveProperty("instrument_ids");
    expect(next).not.toHaveProperty("weights");
  });
  it("class selection clears General", () => {
    const next = changeClass(generalUniverse(explicit), "equity", true);
    expect(next.general).toBe(false);
    expect(next.classes).toEqual(["equity"]);
  });
  it("mode and class changes discard stale custom budgets", () => {
    expect(changeMode(explicit, "liquidity")).not.toHaveProperty("weights");
    expect(changeClass(explicit, "commodity", true)).toMatchObject({
      instrument_ids: [],
      weights: null,
      allocation: "equal",
    });
  });
  it("summarizes volume window and lag", () => {
    expect(marketSummary(generalUniverse(explicit))).toContain("1d lag");
  });
});
