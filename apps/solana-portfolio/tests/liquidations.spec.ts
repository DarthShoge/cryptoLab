import {test, expect} from "@playwright/test";

test("liquidations have red chart markers, a filter, and both collateral/debt legs", async ({page}) => {
  await page.route("**/api/portfolio?*mode=demo*", async route => {
    const data = await (await route.fetch()).json();
    const base = {signature:"same-liquidation",obligation:"loan",time:data.asOf-5*86400+3600,type:"liquidation",price:100,quote:"USD",funding:"unknown",protocol:"Kamino",transactionName:"liquidateObligationAndRedeemReserveCollateralV2"};
    data.trades = [{...base,id:"collateral",asset:"SOL",amount:3,value:300,liquidationRole:"collateral-seized"},{...base,id:"debt",asset:"USDC",amount:280,price:1,value:280,liquidationRole:"debt-repaid"},...data.trades];
    await route.fulfill({json:data});
  });
  await page.goto("/");
  const marker = page.locator(".candlestick-svg").getByRole("button", {name:/1 liquidation events on/});
  await expect(marker).toBeVisible();
  await expect(marker.locator("path")).toHaveAttribute("fill", "#f46d75");
  await marker.click();
  await expect(page.locator(".liquidation-legs")).toContainText("Collateral seized: 3 SOL");
  await expect(page.locator(".liquidation-legs")).toContainText("Debt repaid: 280 USDC");
  await page.locator(".activity-panel").getByRole("button", {name:"Liquidations",exact:true}).click();
  await expect(page.locator(".activity-panel tbody tr")).toHaveCount(2);
  await expect(page.locator(".activity-panel")).toContainText("Collateral seized");
  await expect(page.locator(".activity-panel")).toContainText("Debt repaid");
});
