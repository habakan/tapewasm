import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const expected = JSON.parse(
  readFileSync(resolve(here, "fixtures/advi_expected.json"), "utf8"),
);

test("mean-field ADVI recovers a closed-form posterior in this engine", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto("/advi.html");
  const result = await page.waitForFunction(() => (window as any).result, null, {
    timeout: 60_000,
  }).then((h) => h.jsonValue() as any);

  expect(errors, `page errors: ${errors.join("; ")}`).toEqual([]);
  expect(result.error ?? null, "the page reported an error").toBeNull();
  expect(result.ok).toBe(true);

  // Pure arithmetic imports no `Math.*`, so every engine lands on the same numbers.
  for (let k = 0; k < expected.mean.length; k++) {
    const muGap = Math.abs(result.mu[k] - expected.mean[k]) / expected.sd[k];
    expect(muGap, `mu[${k}]: ${result.mu[k]} vs ${expected.mean[k]}`).toBeLessThan(0.05);
    const sigmaGap = Math.abs(result.sigma[k] - expected.sd[k]) / expected.sd[k];
    expect(sigmaGap, `sigma[${k}]: ${result.sigma[k]} vs ${expected.sd[k]}`).toBeLessThan(0.05);
  }
});
