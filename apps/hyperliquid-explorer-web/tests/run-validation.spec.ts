import { test, expect } from "@playwright/test";

test("coverage is explained before running and rapid clicks save only once", async ({
  page,
}) => {
  let submissions = 0;
  page.on("request", (request) => {
    if (
      request.method() === "POST" &&
      request.url().endsWith("/api/lab/experiments")
    )
      submissions++;
  });
  await page.goto("/");
  const run = page.getByRole("button", {
    name: "Run and save backtest",
    exact: true,
  });
  await expect(
    page.getByText(
      /90-day trader lookback and applicable normalization warmup/,
    ),
  ).toBeVisible();
  await expect(
    page.getByText(/Backtest end 2026-01-08 exceeds dataset coverage/),
  ).toBeVisible();
  await expect(run).toBeDisabled();
  await page.getByRole("button", { name: "Load synthetic preset" }).click();
  await expect(
    page.getByText("Ready to submit", { exact: true }),
  ).toBeVisible();
  await page
    .getByLabel("Hypothesis name", { exact: true })
    .fill("One guarded submission");
  await expect(run).toBeEnabled();
  await run.evaluate((button: HTMLButtonElement) => {
    button.click();
    button.click();
  });
  await expect(
    page.getByRole("heading", { name: "One guarded submission", exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText("completed", { exact: true }).first(),
  ).toBeVisible({ timeout: 20000 });
  expect(submissions).toBe(1);
});

test("obsolete preflight cannot replace the current draft readiness", async ({
  page,
}) => {
  let release!: () => void;
  let observed!: () => void;
  const seen = new Promise<void>((resolve) => {
    observed = resolve;
  });
  const delayed = new Promise<void>((resolve) => {
    release = resolve;
  });
  let first = true;
  await page.route("**/api/lab/preflight", async (route) => {
    if (!first) return route.continue();
    first = false;
    const response = await route.fetch();
    observed();
    await delayed;
    await route.fulfill({ response }).catch(() => {});
  });
  await page.goto("/");
  await seen;
  await page.getByRole("button", { name: "Load synthetic preset" }).click();
  await expect(
    page.getByText("Ready to submit", { exact: true }),
  ).toBeVisible();
  release();
  await expect(
    page.getByRole("button", { name: "Run and save backtest", exact: true }),
  ).toBeEnabled();
  await expect(
    page.getByText(
      /90-day trader lookback and applicable normalization warmup/,
    ),
  ).toHaveCount(0);
});

test("server rejection focuses an actionable error and preserves draft", async ({
  page,
}) => {
  await page.route("**/api/lab/experiments", (route) =>
    route.request().method() === "POST"
      ? route.fulfill({
          status: 422,
          json: {
            detail:
              "Dataset checksum changed since registration; ask the operator to check the dataset.",
          },
        })
      : route.continue(),
  );
  await page.goto("/");
  await page.getByRole("button", { name: "Load synthetic preset" }).click();
  await page
    .getByLabel("Hypothesis name", { exact: true })
    .fill("Preserve this draft");
  await page
    .getByRole("button", { name: "Run and save backtest", exact: true })
    .click();
  const alert = page.getByRole("alert");
  await expect(alert).toContainText("Dataset checksum changed");
  await expect(alert).toBeFocused();
  await expect(page.getByLabel("Hypothesis name", { exact: true })).toHaveValue(
    "Preserve this draft",
  );
});
