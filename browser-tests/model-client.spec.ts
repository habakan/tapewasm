import { test, expect } from "@playwright/test";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const expected = JSON.parse(readFileSync(resolve(here, "fixtures/expected.json"), "utf8"));
const meta = JSON.parse(readFileSync(resolve(here, "fixtures/meta.json"), "utf8"));

test("the model client returns retained, named chains with sampler statistics", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto("/model-client.html");
  await page.waitForFunction(() => typeof (window as any).runSample === "function");
  expect(errors, errors.join("; ")).toEqual([]);

  const result = await page.evaluate(async () => {
    await (window as any).runSample();
    return (window as any).result;
  });

  expect(errors, errors.join("; ")).toEqual([]);
  expect(result.ok).toBe(true);
  expect(result.parameterNames).toEqual(meta.paramNames);
  expect(result.space).toBe("as-compiled");
  expect(result.nParams).toBe(meta.nParams);
  expect(result.nDraws).toBe(meta.draws);
  expect(result.chains).toHaveLength(2);
  expect(result.diverging).toHaveLength(2);
  for (const chain of result.chains) expect(chain).toHaveLength(meta.draws * meta.nParams);
  for (const chain of result.diverging) expect(chain).toHaveLength(meta.draws);

  for (let k = 0; k < expected.mean.length; k++) {
    const values = result.chains.flatMap((chain: number[]) =>
      Array.from({ length: meta.draws }, (_, i) => chain[i * meta.nParams + k]),
    );
    const mean = values.reduce((sum: number, value: number) => sum + value / values.length, 0);
    const gap = Math.abs(mean - expected.mean[k]) / Math.max(expected.sd[k], 1e-12);
    expect(gap, `${meta.paramNames[k]}: ${mean} vs ${expected.mean[k]}`).toBeLessThan(0.3);
  }
});

test("aborting a run retires its worker and a newly loaded model can sample", async ({ page }) => {
  await page.goto("/model-client.html");
  await page.waitForFunction(() => typeof (window as any).cancelSample === "function");
  const state = page.locator("#status");
  const running = page.evaluate(() => (window as any).cancelSample());
  await expect(state).toHaveText("sampling", { timeout: 30_000 });
  await page.getByRole("button", { name: "Cancel" }).click();
  await running;

  const result = await page.evaluate(() => ({
    cancelled: (window as any).cancelResult,
    reloaded: (window as any).reloadResult,
  }));
  expect(result.cancelled).toEqual({ rejected: true, name: "AbortError" });
  expect(result.reloaded).toEqual({ ok: true, nDraws: 50 });
});

test("invalid metadata rejects with a field-specific error", async ({ page }) => {
  await page.goto("/model-client.html");
  const error = await page.evaluate(async () => {
    const { loadModel } = await import("/ts/browser.js");
    try {
      await loadModel({
        wasmUrl: "/fixtures/model.wasm",
        metadataUrl: "/invalid-meta.json",
      });
      return null;
    } catch (error) {
      return String(error);
    }
  });
  expect(error).toContain("nParams");
});
