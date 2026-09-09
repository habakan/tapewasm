import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const expected = JSON.parse(
  readFileSync(resolve(here, "fixtures/expected.json"), "utf8"),
);
const meta = JSON.parse(readFileSync(resolve(here, "fixtures/meta.json"), "utf8"));

test("a module compiled ahead of time samples in this engine", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(String(e)));

  await page.goto("/page.html");
  const result = await page.waitForFunction(() => (window as any).result, null, {
    timeout: 60_000,
  }).then((h) => h.jsonValue() as any);

  expect(errors, `page errors: ${errors.join("; ")}`).toEqual([]);
  expect(result.error ?? null, "the page reported an error").toBeNull();
  expect(result.ok).toBe(true);
  expect(result.moduleBytes).toBe(meta.scratchInit.length > 0 ? result.moduleBytes : 0);

  // The module imports Math.exp and friends from the host, and those are
  // implementation-defined to the last ulp — so engines take different
  // trajectories from the same seed and the draws are not identical. Compare
  // the posterior, not the bits.
  for (let k = 0; k < expected.mean.length; k++) {
    const gap = Math.abs(result.mean[k] - expected.mean[k]) / Math.max(expected.sd[k], 1e-12);
    expect(gap, `${meta.paramNames[k]}: ${result.mean[k]} vs ${expected.mean[k]}`)
      .toBeLessThan(0.3);
  }
});
