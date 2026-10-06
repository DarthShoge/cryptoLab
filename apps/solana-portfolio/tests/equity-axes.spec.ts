import { test, expect } from "@playwright/test";

test("equity axes show units, full dates and actual elapsed time since inception", async ({ page }) => {
  const times = [Date.UTC(2024, 0, 1, 12), Date.UTC(2024, 0, 2, 12), Date.UTC(2026, 0, 1, 12)].map(time => time / 1000);
  await page.route("**/api/portfolio?*mode=demo*", async route => {
    const response = await route.fetch();
    const data = await response.json();
    data.history = times.map((time, index) => ({ time, equity: 1000 * (index + 1), externalFlow: 0, solPrice: 100 }));
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  const panel = page.locator(".performance");
  await panel.getByRole("button", { name: "Inception", exact: true }).click();
  const chart = panel.getByRole("img", { name: "Portfolio equity history" });
  await expect(chart).toBeVisible();
  await expect(chart.locator(".equity-axis-unit")).toHaveText("USD");
  await expect(chart.locator(".equity-y-tick").first()).toContainText("$");
  await expect(chart.locator(".equity-x-tick").first()).toHaveText("01 Jan 2024");
  await expect(chart.locator(".equity-x-tick").last()).toHaveText("01 Jan 2026");
  const coords = await chart.locator(".equity-line").evaluate(path => [...path.getAttribute("d")!.matchAll(/[ML]([\d.]+),/g)].map(match => Number(match[1])));
  expect((coords[1] - coords[0]) / (coords[2] - coords[0])).toBeCloseTo(1 / 731, 4);
  const point = await chart.locator("circle").evaluate(circle => {
    const svg = circle.ownerSVGElement!;
    const pt = svg.createSVGPoint();
    const xs = [...svg.querySelector(".equity-line")!.getAttribute("d")!.matchAll(/[ML]([\d.]+),/g)].map(match => Number(match[1]));
    pt.x = xs[0] + .1 * (xs[2] - xs[0]);
    pt.y = 80;
    const screen = pt.matrixTransform(svg.getScreenCTM()!);
    return { x: screen.x, y: screen.y };
  });
  await page.mouse.move(point.x, point.y);
  await expect(panel.locator(".chart-dates")).toContainText("02 Jan 2024 · 12:00 UTC");
  await panel.getByRole("button", { name: "SOL", exact: true }).click();
  await expect(chart.locator(".equity-axis-unit")).toHaveText("SOL");
  await expect(chart.locator(".equity-y-tick").first()).not.toContainText("$");
  await page.setViewportSize({ width: 390, height: 844 });
  const ticks = chart.locator(".equity-x-tick");
  await expect(ticks).toHaveCount(3);
  const boxes = await ticks.evaluateAll(labels => labels.map(label => { const rect = label.getBoundingClientRect(); return { left: rect.left, right: rect.right }; }));
  for (let i = 1; i < boxes.length; i++) expect(boxes[i].left).toBeGreaterThan(boxes[i - 1].right);
});

test("partial equity coverage changes break the plotted line", async ({ page }) => {
  await page.route("**/api/portfolio?*mode=demo*", async route => {
    const data = await (await route.fetch()).json();
    data.history = [0, 1, 2].map(i => ({ time: 1700000000 + i * 86400, equity: 1000 + i * 100, externalFlow: null, solPrice: 100, valuationComplete: false, unpricedCount: i === 2 ? 3 : 2 }));
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await expect(page.locator(".equity-line")).toBeVisible();
  const path = await page.locator(".equity-line").getAttribute("d");
  expect(path!.match(/M/g)).toHaveLength(2);
  await expect(page.locator(".performance .equity-scope-note")).toContainText("Priced portion only");
});

test("historical Kamino scope shows individual buckets and distinguishes portfolio observations", async ({ page }) => {
  await page.route("**/api/portfolio?*mode=demo*", async route => {
    const data = await (await route.fetch()).json();
    data.mode = "live";
    data.kaminoHistory = [];
    data.kaminoSeries = [{ id: data.summary.loans[0].address, label: "Kamino obligation 1", history: [{ time: 1702857600, equity: 1000, externalFlow: null, solPrice: 100 }, { time: 1735689600, equity: null, externalFlow: null, solPrice: 100 }, { time: 1790640000, equity: 2000, externalFlow: null, solPrice: 100 }] }];
    data.kaminoHistoryWarnings = ["Daily buckets exclude wallet balances."];
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await page.getByRole("button", {name: "Inception", exact: true}).click();
  await expect(page.locator(".performance h2")).toContainText("Kamino net equity");
  await expect(page.locator(".equity-x-tick").first()).toContainText("2023");
  await expect(page.locator(".equity-x-tick").last()).toContainText("2026");
  await expect(page.locator(".performance .equity-scope-note")).toContainText("wallet assets excluded");
  expect((await page.locator(".equity-line").getAttribute("d"))!.match(/M/g)).toHaveLength(2);
  await page.getByLabel("Equity history scope").selectOption("portfolio");
  await expect(page.locator(".performance h2")).toContainText("Known portfolio equity");
});
