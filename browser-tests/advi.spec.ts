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

test("advi's on_snapshot reports each snapshot without changing the fit", async ({ page }) => {
  await page.goto("/advi.html");
  const r = await page.waitForFunction(() => (window as any).result, null, {
    timeout: 60_000,
  }).then((h) => h.jsonValue() as any);
  expect(r.error ?? null).toBeNull();

  expect(r.hookedMu).toEqual(r.mu);
  expect(r.calls.map((c: any) => c.iter)).toEqual(r.snapshotIters);
  const n = r.mu.length;
  r.calls.forEach((c: any, s: number) => {
    expect(c.mu).toEqual(r.muSnapshots.slice(s * n, (s + 1) * n));
  });
  // Each call carries only the trace since the one before, so together they are all of it.
  expect(r.calls.reduce((a: number, c: any) => a + c.elboLen, 0)).toBe(r.elboLen);
});
