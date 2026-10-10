import { expect, test } from "@playwright/test";

test("one compiled model accepts new runtime data in each engine", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  await page.goto("/runtime-data.html");
  const result = await page.waitForFunction(() => (window as any).result, null, {
    timeout: 30_000,
  }).then((handle) => handle.jsonValue() as any);

  expect(errors, `page errors: ${errors.join("; ")}`).toEqual([]);
  expect(result.error ?? null).toBeNull();
  expect(result.nData).toBe(1);
  expect(result.samplerNData).toBe(1);
  expect(result.missingData).toContain("setData");
  expect(result.first[0]).toBeCloseTo(-0.08, 12);
  expect(result.first[1]).toBeCloseTo(0.4, 12);
  expect(result.output[0]).toBeCloseTo(-0.08, 12);
  expect(result.second[0]).toBeCloseTo(-0.5, 12);
  expect(result.second[1]).toBeCloseTo(-1.0, 12);
  expect(result.wrongLength).toContain("expects 1");
});
