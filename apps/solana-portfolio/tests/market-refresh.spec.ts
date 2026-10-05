import {test,expect} from "@playwright/test";

test("prices and labeled refresh stay visible on mobile and refresh requests fresh data",async({page})=>{
  await page.setViewportSize({width:390,height:844});
  await page.goto("/");
  const strip=page.getByRole("region",{name:"Market prices"});
  await expect(strip).toBeVisible();
  for(const symbol of ["ETH","SOL","BTC"]) await expect(strip).toContainText(symbol);
  await expect(strip).toContainText("Demonstration prices");
  await expect(page.locator(".candlestick-svg")).toBeVisible();
  const response=page.waitForResponse(r=>r.url().includes("/api/portfolio?") && new URL(r.url()).searchParams.get("refresh")==="1");
  await page.getByRole("button",{name:"Refresh account",exact:true}).click();
  await response;
  await expect(page.locator(".candlestick-svg")).toBeVisible();
  await expect(page.getByRole("button",{name:"Refresh account",exact:true})).toContainText("Refresh");
  expect(await page.evaluate(()=>document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
