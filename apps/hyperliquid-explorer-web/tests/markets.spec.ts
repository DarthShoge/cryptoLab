import { test, expect } from "@playwright/test";

test("cross-class market selection, saved history and clone", async ({
  page,
}, testInfo) => {
  await page.goto("/");
  await page
    .getByLabel("Local dataset", { exact: true })
    .selectOption("market_demo");
  await page.getByRole("button", { name: "Load synthetic preset" }).click();
  await expect(
    page.getByRole("checkbox", { name: "General", exact: true }),
  ).toBeChecked();
  await expect(
    page.getByRole("checkbox", { name: "Crypto", exact: true }),
  ).not.toBeChecked();
  await page
    .getByLabel("Hypothesis name", { exact: true })
    .fill("Cross-class liquidity copying");
  await page
    .getByRole("button", { name: "Run and save backtest", exact: true })
    .click();
  await expect(
    page.getByText("completed", { exact: true }).first(),
  ).toBeVisible({ timeout: 20000 });
  await page.getByRole("tab", { name: "Market universe", exact: true }).click();
  await expect(
    page.getByRole("heading", { name: "Historical market universe" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "USD volume" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Inspect traders" }).first().click();
  await expect(
    page.getByRole("heading", { name: "Historical trader universe" }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Clone configuration", exact: true })
    .click();
  await page.getByRole("checkbox", { name: "Equities", exact: true }).check();
  await expect(
    page.getByRole("checkbox", { name: "General", exact: true }),
  ).not.toBeChecked();
  await page.getByLabel("Market selection mode").selectOption("explicit");
  await page.getByRole("checkbox", { name: /demo:STOCK/ }).check();
  await expect(
    page.getByRole("checkbox", { name: /other:STOCK/ }),
  ).toBeVisible();
  await page
    .getByLabel("Hypothesis name", { exact: true })
    .fill("Explicit equity copying");
  await page
    .getByRole("button", { name: "Run and save backtest", exact: true })
    .click();
  await expect(
    page.getByText("completed", { exact: true }).first(),
  ).toBeVisible({ timeout: 20000 });
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(page.viewportSize()!.width);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.getByRole("tab", { name: "Market universe", exact: true }).click();
  await expect(
    page.getByRole("columnheader", { name: "USD volume" }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: testInfo.outputPath("market-history-mobile.png"),
    fullPage: true,
  });
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page
    .getByRole("button", { name: "Saved backtests", exact: true })
    .click();
  await page
    .getByRole("checkbox", {
      name: "Compare Cross-class liquidity copying",
      exact: true,
    })
    .check();
  await page
    .getByRole("checkbox", {
      name: "Compare Explicit equity copying",
      exact: true,
    })
    .check();
  await page
    .getByRole("button", { name: "Compare selected", exact: true })
    .click();
  await expect(
    page.getByRole("rowheader", { name: "Market universe", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("Mean market membership turnover", { exact: true }),
  ).toBeVisible();
});
