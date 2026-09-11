// Times one gradient per engine, straight-line ("never") against re-rolled
// ("always"), over tapes sized either side of `RE_ROLL_ABOVE`. Slow and
// machine-dependent, so not in CI:
//
//   make bench                       all three engines
//   node bench.mjs chromium          one; OUT=results.json also writes the rows

import { writeFile } from "node:fs/promises";
import { chromium, firefox, webkit } from "@playwright/test";
import "./serve.mjs";

const CASES = [
  { shape: "linreg", args: [200] },
  { shape: "linreg", args: [300] },
  { shape: "linreg", args: [400] },
  { shape: "linreg", args: [600] },
  { shape: "linreg", args: [1000] },
  { shape: "linreg", args: [1700] },
  { shape: "linreg", args: [5000] },
  { shape: "matvec", args: [500, 10, true] },
  { shape: "matvec", args: [2000, 10, true] },
  { shape: "matvec", args: [5000, 10, true] },
  { shape: "matvec", args: [100, 10, false] },
  { shape: "matvec", args: [200, 10, false] },
  { shape: "matvec", args: [1000, 10, false] },
];

const engines = { chromium, firefox, webkit };
const wanted = (process.argv[2] ?? "chromium,firefox,webkit").split(",");
const rows = [];
for (const name of wanted) {
  const browser = await engines[name].launch();
  const page = await browser.newPage();
  await page.goto("http://127.0.0.1:8123/bench.html");
  await page.waitForFunction(() => window.ready);
  for (const r of await page.evaluate((cases) => window.runBench(cases), CASES)) {
    rows.push({ engine: name, ...r });
  }
  await browser.close();
}

const key = (r) => `${r.shape}(${r.args.join(",")})`;
console.log(`${"tape".padEnd(22)} ${"nodes".padStart(6)}  ${"engine".padEnd(9)} ${"never µs".padStart(9)} ${"always µs".padStart(9)} ${"ratio".padStart(6)}   bytes never / always`);
for (const k of [...new Set(rows.map(key))]) {
  for (const name of wanted) {
    const never = rows.find((r) => key(r) === k && r.engine === name && r.mode === "never");
    const always = rows.find((r) => key(r) === k && r.engine === name && r.mode === "always");
    console.log(
      `${k.padEnd(22)} ${String(never.lines).padStart(6)}  ${name.padEnd(9)} ` +
      `${never.us.toFixed(1).padStart(9)} ${always.us.toFixed(1).padStart(9)} ` +
      `${(always.us / never.us).toFixed(2).padStart(6)}   ${never.bytes} / ${always.bytes}`,
    );
  }
}
if (process.env.OUT) await writeFile(process.env.OUT, JSON.stringify(rows, null, 1));
process.exit(0);
