import { test, expect } from "@playwright/test";

test("historical health strip aligns to candles, uses risk colors and leaves gaps unknown", async ({ page }) => {
  await page.route("**/api/portfolio?*mode=demo*", async route => {
    const data = await (await route.fetch()).json();
    const end = data.asOf;
    const points = [
      { time: end - 3 * 86400, health: .95, status: "available", adjustedDebt: 100, liquidationLimit: 95 },
      { time: end - 2 * 86400, health: 1.5, status: "available", adjustedDebt: 100, liquidationLimit: 150 },
      { time: end - 86400, health: null, status: "no-debt", adjustedDebt: 0, liquidationLimit: 150 },
    ];
    data.healthHistory = points;
    data.kaminoSeries = [{id: data.summary.loans[0].address, label: "Kamino obligation 1", history: [], healthHistory: points}];
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  const strip = page.getByRole("img", {name: "Historical Kamino health heatmap"});
  await expect(strip).toBeVisible();
  await expect(strip.locator('rect[data-health-status="available"]')).toHaveCount(2);
  await expect(strip.locator('rect[data-health-status="no-debt"]')).toHaveCount(1);
  expect(await strip.locator('rect[data-health-status="unavailable"]').count()).toBeGreaterThan(1);
  const red = strip.locator('rect[data-health-ratio="0.95"]');
  const green = strip.locator('rect[data-health-ratio="1.5"]');
  await expect(red).toHaveAttribute("fill", "hsl(0 72% 52%)");
  await expect(green).toHaveAttribute("fill", "hsl(140 72% 52%)");
  const offset = await red.evaluate(rect => {
    const time = rect.getAttribute("data-candle-time");
    const wick = document.querySelector(`.candlestick-svg g[data-candle-time="${time}"] line`)! as SVGLineElement;
    const candlePoint = wick.ownerSVGElement!.createSVGPoint();
    candlePoint.x = wick.x1.baseVal.value; candlePoint.y = 0;
    const heatRect = rect as SVGRectElement;
    const heatPoint = heatRect.ownerSVGElement!.createSVGPoint();
    heatPoint.x = heatRect.x.baseVal.value + heatRect.width.baseVal.value / 2; heatPoint.y = 0;
    return Math.abs(candlePoint.matrixTransform(wick.getScreenCTM()!).x - heatPoint.matrixTransform(rect.getScreenCTM()!).x);
  });
  expect(offset).toBeLessThan(1);
  await red.hover();
  await expect(page.locator(".health-heatmap-detail")).toContainText("0.950×");
  await expect(page.locator(".health-heatmap-detail")).toContainText("Daily bucket");
  await page.getByRole("button", {name: "1H", exact: true}).click();
  await expect(strip.locator('rect[data-health-status="no-debt"]')).toHaveCount(24);
  await strip.locator('rect[data-health-status="no-debt"]').last().hover();
  await expect(page.locator(".health-heatmap-detail")).toContainText("No debt");
  const time = await strip.locator('rect[data-health-status="no-debt"]').last().getAttribute("data-candle-time");
  await page.getByRole("button", {name: "Zoom in", exact: true}).click();
  await expect(strip.locator(`rect[data-candle-time="${time}"]`)).toBeVisible();
});

test("missing current obligation history never borrows another obligation's values", async ({page}) => {
  await page.route("**/api/portfolio?*mode=demo*", async route => {
    const data = await (await route.fetch()).json();
    data.mode = "live";
    data.summary.loans[0].address = "current-without-history";
    const points = [{time:data.asOf-2*86400,health:2,status:"available"},{time:data.asOf-86400,health:2,status:"available"}];
    data.healthHistory = points;
    data.kaminoSeries = [{id:"historical-other",label:"Historical obligation",healthHistory:points,history:[{time:data.asOf-2*86400,equity:1000,externalFlow:null,solPrice:100},{time:data.asOf-86400,equity:2000,externalFlow:null,solPrice:100}]}];
    await route.fulfill({json:data});
  });
  await page.goto("/");
  const strip=page.getByRole("img", {name:"Historical Kamino health heatmap"});
  await expect(strip).toBeVisible();
  await expect(strip.locator('rect[data-health-status="available"]')).toHaveCount(0);
  await expect(page.locator(".performance .empty-chart")).toBeVisible();
  await page.getByLabel("Historical health obligation").selectOption("all");
  await expect(strip.locator('rect[data-health-status="available"]')).toHaveCount(2);
});
