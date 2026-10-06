import { test, expect } from "@playwright/test";

test("Orca SOL quote legs and Raydium token swaps show their actual buy/sell sides", async ({page}) => {
  await page.route("**/api/portfolio?*mode=demo*", async route => {
    const data = await (await route.fetch()).json();
    const time = data.asOf - 3 * 86400 + 3600;
    data.trades = [
      {id:"orca-test",signature:null,time,type:"buy",asset:"TEST",amount:10,price:.2,value:2,quote:"SOL",funding:"unknown",protocol:"Orca",venues:["Orca"], executionLegs:[{mint:"test",asset:"TEST",side:"buy",amount:10,price:.2,quote:"SOL",quoteMint:"sol",value:2},{mint:"sol",asset:"SOL",side:"sell",amount:2,price:5,quote:"TEST",quoteMint:"test",value:10}]},
      {id:"raydium-test",signature:null,time:time+86400,type:"swap",asset:null,amount:null,price:null,value:null,quote:null,funding:"unknown",protocol:"Raydium",venues:["Raydium"],executionLegs:[{mint:"btc",asset:"BTC",side:"sell",amount:.1,price:20,quote:"ETH",quoteMint:"eth",value:2},{mint:"eth",asset:"ETH",side:"buy",amount:2,price:.05,quote:"BTC",quoteMint:"btc",value:.1}]},
    ];
    await route.fulfill({json:data});
  });
  await page.goto("/");
  const orcaMarker = page.locator(".candlestick-svg").getByRole("button", {name:/sell 2 SOL.*via Orca/});
  await expect(orcaMarker).toBeVisible();
  await orcaMarker.click();
  await expect(page.locator(".execution-legs")).toContainText("SELL 2 SOL");
  await expect(page.locator(".execution-legs")).toContainText("0.2 SOL per TEST");
  await expect(page.getByLabel("Chart asset")).toHaveValue("SOL");
  await page.getByLabel("Chart asset").selectOption("ETH");
  await expect(page.locator(".candlestick-svg").getByRole("button", {name:/buy 2 ETH.*via Raydium/})).toBeVisible();
  await page.locator(".activity-panel").getByRole("button", {name:"DEX trades",exact:true}).click();
  await expect(page.locator(".activity-panel tbody tr")).toHaveCount(2);
});
