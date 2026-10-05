import {test,expect} from "@playwright/test";

test("Supertrend does not connect across warmup or missing candle periods",async({page})=>{
  await page.route("**/api/chart?*",async route=>{
    const data=await(await route.fetch()).json();
    data.candles=data.candles.slice(0,6);
    data.candles[4].time+=86400;
    data.candles[4].endTime+=86400;
    data.candles[5].time+=86400;
    data.candles[5].endTime+=86400;
    data.indicator=data.candles.map((c:any,i:number)=>({time:c.time,value:i===1?null:c.close,direction:i===1?null:"bullish"}));
    await route.fulfill({json:data});
  });
  await page.goto("/");
  await expect(page.locator(".supertrend-line")).toHaveCount(3);
  const lengths=await page.locator(".supertrend-line").evaluateAll(elements=>elements.map(element=>element.getAttribute("points")!.split(" ").length));
  expect(lengths).toEqual([1,2,2]);
});

test("equity checkbox adds independent axis and daily weekly monthly labels are clear", async({page})=>{
  await page.goto("/");
  await expect(page.getByRole("button",{name:"1 Day",exact:true})).toBeVisible();
  await page.getByRole("checkbox",{name:"Overlay equity",exact:true}).check();
  await expect(page.locator(".chart-equity-axis")).toContainText("Equity USD");
  await expect(page.locator(".chart-equity-line")).toBeVisible();
  const priceTicks=await page.locator(".chart-price-tick").allTextContents();
  await page.getByRole("checkbox",{name:"Overlay equity",exact:true}).uncheck();
  await expect(page.locator(".chart-equity-line")).toHaveCount(0);
  expect(await page.locator(".chart-price-tick").allTextContents()).toEqual(priceTicks);
  await page.getByRole("button",{name:"1 Week",exact:true}).click();
  await expect(page.locator(".timeframe-caption")).toContainText("1 week");
  await expect(page.locator(".candlestick-svg")).toBeVisible();
  await page.getByRole("button",{name:"1 Month",exact:true}).click();
  await expect(page.locator(".timeframe-caption")).toContainText("calendar month");
  await expect(page.locator(".candlestick-svg")).toBeVisible();
});

test("combined trend uses red green gradient states and unknown warmup",async({page})=>{
  await page.route("**/api/chart?*",async route=>{
    const data=await(await route.fetch()).json();
    if(new URL(route.request().url()).searchParams.get("combined")==="1") {
      data.combined=data.candles.map((c:any,i:number)=>({time:c.time,score:i===0?null:([-1,-1/3,1/3,1] as number[])[i%4],bullishCount:i%4,bearishCount:3-i%4,directions:{"1d":i%4>0?"bullish":"bearish","1w":i%4>1?"bullish":"bearish","1M":i%4>2?"bullish":"bearish"}}));
    }
    await route.fulfill({json:data});
  });
  await page.goto("/");
  await page.getByRole("button",{name:"Combined Supertrend",exact:true}).click();
  await expect(page.locator('.combined-cell[data-score="-1"]').first()).toHaveAttribute("fill","#8f1d32");
  await expect(page.locator('.combined-cell[data-score="1"]').first()).toHaveAttribute("fill","#087a4d");
  await expect(page.locator('.combined-cell[data-score="unknown"]')).toHaveCount(1);
  await expect(page.locator(".combined-status")).toContainText("Daily");
  await expect(page.locator(".combined-status")).toContainText("Weekly");
  await expect(page.locator(".combined-status")).toContainText("Monthly");
});

test("combined and equity controls fit on mobile with actual demo signals",async({page})=>{
  await page.setViewportSize({width:390,height:844});
  await page.goto("/");
  await page.getByRole("checkbox",{name:"Overlay equity",exact:true}).check();
  await page.getByRole("button",{name:"Combined Supertrend",exact:true}).click();
  await expect(page.locator('.combined-cell:not([data-score="unknown"])').first()).toBeVisible();
  await expect(page.locator(".chart-equity-line").first()).toBeVisible();
  const dimensions=await page.evaluate(()=>({width:document.documentElement.scrollWidth,viewport:window.innerWidth}));
  expect(dimensions.width).toBeLessThanOrEqual(dimensions.viewport);
  await page.locator(".trading-chart").screenshot({path:"/tmp/solana-portfolio-combined-equity-mobile.png"});
});
