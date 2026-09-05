import { test, expect } from "@playwright/test";

test("configure, preview, save, clone, compare and inspect historical traders", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Hyperliquid copy-strategy lab" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Load synthetic preset" }).click();
  await page
    .getByLabel("Hypothesis name", { exact: true })
    .fill("Top two SOL and ETH");
  await page
    .getByRole("button", { name: "Preview trader universe", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Historical cohort preview" }),
  ).toBeVisible();
  await expect(page.getByText("Preview complete", { exact: true })).toBeVisible(
    { timeout: 20000 },
  );
  await page
    .getByRole("button", { name: "Run and save backtest", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Top two SOL and ETH", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("completed", { exact: true }).first(),
  ).toBeVisible({ timeout: 20000 });
  await expect(
    page
      .getByRole("combobox", { name: "Scenario", exact: true })
      .locator("option:checked"),
  ).toContainText("Top two SOL and ETH — ETH + SOL");
  await expect(
    page
      .getByRole("combobox", { name: "Scenario", exact: true })
      .locator("option:checked"),
  ).toContainText("equal-weight direction copying");
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.getByRole("tab", { name: "Trader universe", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Historical trader universe" }),
  ).toBeVisible();
  await page.getByLabel("Decision date", { exact: true }).fill("2026-01-04");
  await expect(
    page.getByRole("columnheader", { name: "Percentiles" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: /Inspect wallet/ })
    .first()
    .click();
  await expect(
    page.getByRole("heading", { name: "Wallet membership history" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Clone configuration", exact: true })
    .click();
  await page
    .getByLabel("Hypothesis name", { exact: true })
    .fill("Top three SOL and ETH");
  await page.getByLabel("Top N traders", { exact: true }).fill("3");
  await page
    .getByRole("button", { name: "Run and save backtest", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Top three SOL and ETH", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("completed", { exact: true }).first(),
  ).toBeVisible({ timeout: 20000 });
  await page
    .getByRole("button", { name: "Saved backtests", exact: true })
    .click();
  await page
    .getByRole("checkbox", { name: "Compare Top two SOL and ETH" })
    .check();
  await page
    .getByRole("checkbox", { name: "Compare Top three SOL and ETH" })
    .check();
  await page
    .getByRole("button", { name: "Compare selected", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Configuration differences" }),
  ).toBeVisible();
  await expect(
    page.getByRole("rowheader", { name: "Top n", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Saved backtest equity" }),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("lab-comparison.png"),
    fullPage: true,
  });
  await page.reload();
  await page
    .getByRole("button", { name: "Saved backtests", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Top two SOL and ETH", exact: true }),
  ).toBeVisible();
  expect(errors).toEqual([]);
});

test("strategy builder is usable on mobile and handles no datasets", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "Trader universe & selection" }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: testInfo.outputPath("lab-builder-mobile.png"),
    fullPage: true,
  });
  await page.route("**/api/lab/datasets", (r) => r.fulfill({ json: [] }));
  await page.reload();
  await expect(
    page.getByText("No registered local datasets", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByRole("button", { name: "Run and save backtest", exact: true }),
  ).toBeDisabled();
});
