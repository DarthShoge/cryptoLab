import { test, expect } from "@playwright/test";
test("read-only explorer renders a real API report", async ({
  page,
}, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/reports");
  await expect(
    page.getByRole("heading", { name: "Research explorer" }),
  ).toBeVisible();
  await expect(page.getByText("SYNTHETIC DEMO", { exact: true })).toBeVisible();
  await expect(page.getByText("Sharpe", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("N/A").first()).toBeVisible();
  await expect(
    page.getByRole("heading", { name: "Equity vs benchmark" }),
  ).toBeVisible();
  await page.screenshot({
    path: testInfo.outputPath("overview-desktop.png"),
    fullPage: true,
  });
  await page.getByText("Portfolio analysis", { exact: false }).first().click();
  await expect(page.getByRole("rowheader", { name: /^Calmar/ })).toBeVisible();
  await page.getByText("Choose metric columns", { exact: true }).click();
  await expect(
    page.getByRole("checkbox", { name: "Net pnl", exact: true }),
  ).toHaveCount(0);
  await page.getByLabel("Execution delay").selectOption("15");
  await page.getByRole("tab", { name: "Traders" }).click();
  await expect(
    page.getByRole("heading", { name: "Trader rankings" }),
  ).toBeVisible();
  await expect(
    page.getByRole("columnheader", { name: "Decision time" }).first(),
  ).toBeVisible();
  await page.getByLabel("Wallet search").fill("00005");
  await expect(
    page
      .locator("section")
      .filter({ has: page.getByRole("heading", { name: "Trader rankings" }) })
      .getByText("1 rows", { exact: true }),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Execution" }).click();
  await expect(
    page.getByRole("heading", { name: "Simulated fills" }),
  ).toBeVisible();
  await expect(
    page.getByRole("cell", { name: "partial_depth", exact: true }).first(),
  ).toBeVisible();
  await page
    .getByRole("combobox", { name: /^Asset/ })
    .first()
    .selectOption("ETH");
  await expect(
    page.getByText("No matching rows.", { exact: false }).first(),
  ).toBeVisible();
  await page.getByRole("tab", { name: "Data" }).click();
  await expect(page.getByRole("link", { name: "report.md" })).toBeVisible();
  expect(errors).toEqual([]);
});

test("empty and malformed inventories are explicit", async ({ page }) => {
  await page.route("**/api/runs?*", (route) => route.fulfill({ json: [] }));
  await page.goto("/reports");
  await expect(
    page.getByRole("heading", { name: "No available reports" }),
  ).toBeVisible();
  await page.route("**/api/runs?*", (route) =>
    route.fulfill({
      json: [
        {
          id: "hyperliquid_trader_ensemble_bad",
          available: false,
          warnings: ["Report unavailable or malformed"],
        },
      ],
    }),
  );
  await page.getByRole("button", { name: "Refresh reports" }).click();
  await expect(
    page.getByRole("option", { name: "bad · unavailable" }),
  ).toHaveJSProperty("disabled", true);
});

test("loading and recoverable API errors", async ({ page }) => {
  let release!: () => void;
  const gate = new Promise<void>((resolve) => {
    release = resolve;
  });
  await page.route("**/api/runs?*", async (route) => {
    await gate;
    await route.fulfill({
      status: 503,
      json: { detail: "Temporarily unavailable" },
    });
  });
  await page.goto("/reports");
  await expect(page.getByRole("status").first()).toBeVisible();
  release();
  await expect(page.getByRole("alert")).toBeVisible();
  await page.unroute("**/api/runs?*");
  await page.getByRole("button", { name: "Try again" }).click();
  await expect(
    page.getByRole("heading", { name: "Research explorer" }),
  ).toBeVisible();
});

test("mobile layout keeps the document within the viewport", async ({
  page,
}, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/reports");
  await expect(
    page.getByRole("heading", { name: "Equity vs benchmark" }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
  await page.screenshot({
    path: testInfo.outputPath("overview-mobile.png"),
    fullPage: true,
  });
  await page.getByRole("tab", { name: "Traders" }).click();
  await expect(
    page.getByRole("heading", { name: "Trader rankings" }),
  ).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth),
  ).toBeLessThanOrEqual(390);
});
