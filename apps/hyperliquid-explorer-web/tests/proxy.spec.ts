import { test, expect } from "@playwright/test";

for (const dataset of ["scheduled", "annual"]) {
test(`${dataset}: weekly selection and holdings survive save and clone into a daily comparison`, async ({ page }) => {
  if (dataset === "annual") test.setTimeout(180000);
  const runTimeout = dataset === "annual" ? 90000 : 20000;
  const weeklyName = `${dataset} Weekly baseline`, dailyName = `${dataset} Daily comparison`;
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByLabel("Local dataset").selectOption(dataset);
  await page.getByRole("button", { name: "Load synthetic preset", exact: true }).click();
  await expect(page.getByLabel("Portfolio rebalance cadence")).toHaveValue("weekly");
  for (const name of ["Universe reselection", "Market reselection"]) {
    await expect(page.getByLabel(name, { exact: true })).toHaveValue("weekly");
    await expect(page.getByLabel(name, { exact: true })).toBeDisabled();
  }
  await expect(page.getByLabel("Valuation cadence")).toHaveValue("Hourly (60 minutes)");
  await page.getByLabel("Hypothesis name", { exact: true }).fill(weeklyName);
  await page.getByRole("button", { name: "Run and save backtest", exact: true }).click();
  await expect(page.getByText("completed", { exact: true }).first()).toBeVisible({ timeout: runTimeout });
  await expect(page.getByText(/Net simple UTC hourly equity returns/)).toBeAttached();
  await page.getByRole("tab", { name: "Market universe", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Market selection drilldown" })).toBeVisible();
  await page.getByRole("button", { name: "Clone configuration", exact: true }).click();
  await expect(page.getByLabel("Portfolio rebalance cadence")).toHaveValue("weekly");
  await page.getByLabel("Portfolio rebalance cadence").selectOption("daily");
  await expect(page.getByLabel("Universe reselection", { exact: true })).toHaveValue("daily");
  await expect(page.getByLabel("Market reselection", { exact: true })).toHaveValue("daily");
  await page.getByLabel("Hypothesis name", { exact: true }).fill(dailyName);
  await page.getByRole("button", { name: "Run and save backtest", exact: true }).click();
  await expect(page.getByText("completed", { exact: true }).first()).toBeVisible({ timeout: runTimeout });
  await page.getByRole("button", { name: "Saved backtests", exact: true }).click();
  await page.getByRole("checkbox", { name: `Compare ${weeklyName}`, exact: true }).check();
  await page.getByRole("checkbox", { name: `Compare ${dailyName}`, exact: true }).check();
  await page.getByRole("button", { name: "Compare selected", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Saved backtest equity" })).toBeVisible();
  const cadenceRow = page.getByRole("row").filter({ has: page.getByRole("rowheader", { name: "Rebalance", exact: true }) });
  await expect(cadenceRow.getByRole("cell", { name: '"weekly"', exact: true })).toBeVisible();
  await expect(cadenceRow.getByRole("cell", { name: '"daily"', exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});
}

test("hourly proxy settings survive save, clone, and comparison", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByLabel("Local dataset").selectOption("proxy");
  await page.getByRole("button", { name: "Load synthetic preset", exact: true }).click();
  await expect(page.getByText("APPROXIMATE HOURLY PROXY", { exact: true })).toBeVisible();
  await expect(page.getByLabel("Target update cadence (minutes)")).toHaveValue("Hourly (60 minutes)");
  await page.getByLabel("Hypothesis name", { exact: true }).fill("Hourly baseline");
  await page.getByLabel("Proxy slippage (basis points)").fill("7");
  await page.getByRole("button", { name: "Run and save backtest", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Hourly baseline", exact: true })).toBeVisible();
  await expect(page.getByText("completed", { exact: true }).first()).toBeVisible({ timeout: 20000 });
  await expect(page.getByText(/Net simple UTC hourly equity returns/)).toBeAttached();
  await page.getByRole("tab", { name: "Market universe", exact: true }).click();
  await expect(page.getByRole("heading", { name: "Market selection drilldown" })).toBeVisible();
  await expect(page.getByText("Proxy: ETHUSDT", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Clone configuration", exact: true }).click();
  await expect(page.getByLabel("Proxy slippage (basis points)")).toHaveValue("7");
  await page.getByLabel("Hypothesis name", { exact: true }).fill("Hourly higher slippage");
  await page.getByLabel("Proxy slippage (basis points)").fill("25");
  await page.getByRole("button", { name: "Run and save backtest", exact: true }).click();
  await expect(page.getByText("completed", { exact: true }).first()).toBeVisible({ timeout: 20000 });
  await page.getByRole("button", { name: "Saved backtests", exact: true }).click();
  await page.getByRole("checkbox", { name: "Compare Hourly baseline", exact: true }).check();
  await page.getByRole("checkbox", { name: "Compare Hourly higher slippage", exact: true }).check();
  await page.getByRole("button", { name: "Compare selected", exact: true }).click();
  await expect(page.getByText("Different proxy execution/mark assumptions", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Saved backtest equity" })).toBeVisible();
  expect(errors).toEqual([]);
});
