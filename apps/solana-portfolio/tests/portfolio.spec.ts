import { test, expect } from "@playwright/test";

test("overview combines equity, health, candles and clickable executions", async ({ page }) => {
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Portfolio overview" })).toBeVisible();
  await expect(page.getByText("Fictional demonstration", { exact: true })).toBeVisible();
  await expect(page.getByRole("heading", { name: "Kamino health" })).toBeVisible();
  await expect(page.getByRole("img", { name: "SOL candlestick chart with Supertrend and execution markers" })).toBeVisible();
  await page.getByRole("button", { name: /^buy .*SOL on/i }).first().click();
  await expect(page.getByRole("region", { name: "Execution details" })).toBeVisible();
  await expect(page.getByText("Stablecoin-funded long", { exact: true })).toBeVisible();
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.screenshot({ path: "/tmp/solana-portfolio-desktop.png", fullPage: true });
});

test("chart settings, asset selection and activity search work", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Trading analytics", exact: true }).click();
  await page.getByLabel("Chart asset").selectOption("ETH");
  await expect(page.getByRole("img", { name: "ETH candlestick chart with Supertrend and execution markers" })).toBeVisible();
  await page.getByRole("button", { name: "Indicator settings" }).click();
  await page.getByLabel("ATR period").fill("14");
  await expect(page.getByText(/ATR 14 × 3/)).toBeVisible();
  await page.getByRole("button", { name: "Supertrend", exact: true }).click();
  await expect(page.getByRole("button", { name: "Supertrend", exact: true })).toHaveAttribute("aria-pressed", "false");
  await page.getByRole("button", { name: "Activity", exact: true }).click();
  await page.getByLabel("Search activity").fill("cover short");
  await expect(page.getByRole("button", { name: /Buy ETH/ })).toBeVisible();
  await expect(page.getByRole("button", { name: /Buy SOL/ })).toHaveCount(0);
});

test("failed live requests do not present demo account values", async ({ page }) => {
  await page.route("**/api/portfolio?*mode=live*", route => route.fulfill({ status: 502, contentType: "application/json", body: JSON.stringify({ error: "RPC unavailable for test" }) }));
  await page.goto("/");
  await page.getByRole("button", { name: "Load my wallet" }).click();
  await page.getByLabel("Wallet address").fill("11111111111111111111111111111111");
  await page.getByRole("button", { name: "Load live account" }).click();
  await expect(page.getByText("RPC unavailable for test")).toBeVisible();
  await expect(page.getByText("Fictional demonstration")).toHaveCount(0);
  await expect(page.getByRole("heading", { name: "Kamino health" })).toHaveCount(0);
});

test("mobile layout fits viewport and imported history clears demo positions", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/");
  await expect(page.getByRole("heading", { name: "Portfolio overview" })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: "/tmp/solana-portfolio-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Data & settings", exact: true }).click();
  await page.getByLabel("Import portfolio history").setInputFiles({ name: "history.json", mimeType: "application/json", buffer: Buffer.from(JSON.stringify({ source: { name: "Test export", account: "Test account", equityScope: "wallet-and-kamino", cashFlowCoverage: "complete" }, complete: true, flowTiming: "period-end", history: [{ time: 1700000000, equity: 1000, externalFlow: 0, solPrice: 100 }, { time: 1700086400, equity: 1600, externalFlow: 500, solPrice: 100 }], trades: [] })) });
  await expect(page.getByText("Imported history", { exact: true }).first()).toBeVisible();
  await page.getByRole("button", { name: "Close data settings" }).click();
  await expect(page.getByText("$1,600.00").first()).toBeVisible();
  await expect(page.getByText("No Kamino obligation loaded")).toBeVisible();
});

test("live supplied wallet renders actual health and current market chart", async ({ page }) => {
  test.skip(!process.env.PORTFOLIO_LIVE_TEST, "Opt-in read-only mainnet check");
  test.setTimeout(180000);
  await page.goto("/");
  await page.getByRole("button", { name: "Load my wallet" }).click();
  await expect(page.getByLabel("Wallet address")).not.toHaveValue("");
  await page.getByRole("button", { name: "Load live account" }).click();
  await expect(page.locator(".equity-card .metric-label")).toContainText("Known portfolio equity", { timeout: 150000 });
  await expect(page.getByRole("heading", { name: "Kamino health" })).toBeVisible();
  await expect(page.locator(".health-number")).not.toContainText("—");
  await expect(page.getByRole("img", { name: "SOL candlestick chart with Supertrend and execution markers" })).toBeVisible({ timeout: 20000 });
  await expect(page.getByText("Fictional demonstration", { exact: true })).toHaveCount(0);
  await expect(page.getByLabel("Equity history scope")).toHaveValue("current-kamino");
  await expect(page.locator(".performance h2")).toContainText("Kamino net equity");
  await expect(page.locator(".equity-x-tick").first()).toContainText("2023");
  await expect(page.locator(".history-progress")).toContainText("26/08/2021");
  const counts = (await page.locator(".chart-coverage").innerText()).match(/(\d+) of (\d+) candles/)!;
  expect(Number(counts[1])).toBeGreaterThan(1000);
  expect(counts[1]).toBe(counts[2]);
  const healthStrip = page.getByRole("img", {name: "Historical Kamino health heatmap"});
  await expect(healthStrip).toBeVisible();
  expect(await healthStrip.locator('rect[data-health-status="available"]').count()).toBeGreaterThan(500);
  const liquidationMarkers = page.locator(".candlestick-svg").getByRole("button", {name: /liquidation events on/});
  expect(await liquidationMarkers.count()).toBeGreaterThan(0);
  await liquidationMarkers.first().click();
  await expect(page.locator(".liquidation-legs")).toContainText("Collateral seized:");
  await expect(page.locator(".liquidation-legs")).toContainText("Debt repaid:");
  await page.getByRole("button", {name:"Close execution details",exact:true}).click();
  await page.getByRole("button", {name:"Close chart transactions",exact:true}).click();
  await page.getByRole("checkbox", {name:"Overlay equity",exact:true}).check();
  await expect(page.locator(".chart-equity-line").first()).toBeVisible();
  const combinedResponse = page.waitForResponse(response => response.url().includes("/api/chart?") && response.url().includes("combined=1"), {timeout: 60000});
  await page.getByRole("button", {name:"Combined Supertrend",exact:true}).click();
  const combinedData = await (await combinedResponse).json();
  expect(combinedData.combined.length).toBe(combinedData.candles.length);
  expect(combinedData.combined.filter((point: any) => point.score != null).length).toBeGreaterThan(500);
  await expect(page.locator(".combined-cell").first()).toBeVisible();
  await page.getByRole("button", {name:"1 Week",exact:true}).click();
  await expect(page.locator(".candlestick-svg")).toBeVisible({timeout: 60000});
  await page.getByRole("button", {name:"1 Month",exact:true}).click();
  await expect(page.locator(".candlestick-svg")).toBeVisible({timeout: 60000});
  await expect(page.locator(".chart-equity-line").first()).toBeVisible();
  await page.screenshot({ path: "/tmp/solana-portfolio-live.png", fullPage: true });
});

test("large wallets have a compact overview and full searchable positions view", async ({ page }) => {
  await page.route("**/api/portfolio?*mode=demo*", async route => {
    const response = await route.fetch();
    const data = await response.json();
    data.summary.positions = Array.from({ length: 25 }, (_, i) => ({ id: `bulk-${i}`, symbol: `TOKEN${i}`, amount: i + 1, price: null, value: null, kind: "wallet", obligation: null, apy: null }));
    await route.fulfill({ json: data });
  });
  await page.goto("/");
  await expect(page.locator(".positions-panel tbody tr")).toHaveCount(8);
  await page.getByRole("button", { name: "View all 25 positions" }).click();
  await expect(page.getByRole("heading", { name: "Positions & exposure" })).toBeVisible();
  await expect(page.locator(".positions-panel tbody tr")).toHaveCount(25);
  await page.getByLabel("Search positions").fill("TOKEN24");
  await expect(page.locator(".positions-panel tbody tr")).toHaveCount(1);
});

test("inception price chart includes lending and other wallet activity", async ({ page }) => {
  await page.goto("/");
  await page.getByRole("button", { name: "Inception", exact: true }).click();
  await expect(page.locator(".chart-coverage")).toContainText("10 of 10 indexed transactions in view");
  await page.locator(".candlestick-svg").getByRole("button", { name: /wallet transactions on.*borrow USDC/ }).click();
  await expect(page.locator(".chart-activity-list")).toContainText("BORROW USDC");
  await page.locator(".chart-activity-list").getByRole("button", { name: /BORROW USDC/ }).click();
  await expect(page.locator(".trade-detail")).toContainText("borrow", { ignoreCase: true });
});
