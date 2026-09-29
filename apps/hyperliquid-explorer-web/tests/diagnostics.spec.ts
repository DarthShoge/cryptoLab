import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { join } from "node:path";

test("saved strategy diagnostics and cohort drilldown", async ({ page }, testInfo) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto("/");
  await page.getByLabel("Local dataset").selectOption("demo");
  await page.getByRole("button", { name: "Load synthetic preset" }).click();
  await page.getByLabel("Hypothesis name", { exact: true }).fill("Diagnostic test");
  await page.getByRole("button", { name: "Run and save backtest", exact: true }).click();
  await expect(page.getByText("completed", { exact: true }).first()).toBeVisible({ timeout: 30000 });
  await page.getByRole("tab", { name: "Diagnostics", exact: true }).click();
  await expect(page.getByRole("heading", { name: "System card", exact: true })).toBeVisible();
  await expect(page.getByText("Not research-qualified", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Return profiles", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Weekly statistical analysis", exact: true })).toBeVisible();
  await expect(page.getByText("At least 20 complete weeks are required.").first()).toBeVisible();
  await expect(page.getByRole("heading", { name: "Accounting & execution", exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Historical cohort composition", exact: true })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath("diagnostics-desktop.png"), fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: testInfo.outputPath("diagnostics-mobile.png"), fullPage: true });
  await page.getByRole("button", { name: /Inspect cohort/ }).first().click();
  await expect(page.getByRole("heading", { name: "Historical trader universe", exact: true })).toBeVisible();
  expect(errors).toEqual([]);
});

test("diagnostics exposes loading and safe retry", async ({ page }) => {
  await page.goto("/");
  await page.getByLabel("Local dataset").selectOption("demo");
  await page.getByRole("button", { name: "Load synthetic preset" }).click();
  await page.getByRole("button", { name: "Run and save backtest", exact: true }).click();
  await expect(page.getByText("completed", { exact: true }).first()).toBeVisible({ timeout: 30000 });
  let release: () => void = () => {};
  const gate = new Promise<void>((resolve) => { release = resolve; });
  await page.route("**/diagnostics", async (route) => {
    await gate;
    await route.fulfill({ status: 422, contentType: "application/json", body: JSON.stringify({ detail: "Diagnostic fixture unavailable" }) });
  });
  await page.getByRole("tab", { name: "Diagnostics", exact: true }).click();
  await expect(page.getByRole("status", { name: "Loading report data" })).toBeVisible();
  release();
  await expect(page.getByText("Diagnostic fixture unavailable")).toBeVisible();
  await expect(page.getByRole("button", { name: "Try again" })).toBeVisible();
});

test("actual annual weekly artifact renders without a live coordinator", async ({ page }) => {
  const directory = process.env.HL_DIAGNOSTICS_EVIDENCE;
  test.skip(!directory, "Set HL_DIAGNOSTICS_EVIDENCE to an exported annual diagnostics directory");
  const read = (name: string) => JSON.parse(readFileSync(join(directory!, name), "utf8"));
  const data = read("diagnostics.json"), experiment = read("experiment.json"), report = read("report.json");
  const errors: string[] = [];
  const writes: string[] = [];
  page.on("request", request => {
    if (request.url().includes("/api/lab/") && request.method() !== "GET") writes.push(request.url());
  });
  page.on("pageerror", e => errors.push(e.message));
  await page.route("**/api/lab/experiments?*", route => route.fulfill({ json: [experiment] }));
  await page.route(`**/api/lab/experiments/${experiment.id}`, route => route.fulfill({ json: experiment }));
  await page.route(`**/api/runs/${experiment.run_id}`, route => route.fulfill({ json: report }));
  await page.route(`**/api/lab/experiments/${experiment.id}/diagnostics`, route => route.fulfill({ json: data }));
  await page.goto(`/?experiment=${experiment.id}&tab=diagnostics`);
  await expect(page.getByRole("tab", { name: "Diagnostics", exact: true })).toHaveAttribute("aria-selected", "true");
  await expect(page.getByRole("heading", { name: "System card", exact: true })).toBeVisible();
  await expect(page.getByText("$8,368.01", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("-16.32%", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("21.33%", { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("row", { name: "Complete weeks 52", exact: true })).toBeVisible();
  await expect(page.getByText("smoke only unreconciled", { exact: true })).toBeVisible();
  await expect(page.getByRole("button", { name: /Inspect cohort/ })).toHaveCount(117);
  await page.screenshot({ path: join(directory!, "weekly-diagnostics-desktop.png"), fullPage: true });
  await page.getByRole("heading", { name: "Return profiles", exact: true }).scrollIntoViewIfNeeded();
  await page.screenshot({ path: join(directory!, "weekly-return-profiles.png") });
  await page.setViewportSize({ width: 390, height: 844 });
  await expect.poll(() => page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
  await page.screenshot({ path: join(directory!, "weekly-diagnostics-mobile.png"), fullPage: true });
  expect(errors).toEqual([]);
  expect(writes).toEqual([]);
  await page.reload();
  await expect(page.getByRole("heading", { name: "System card", exact: true })).toBeVisible();
  expect(writes).toEqual([]);
});
