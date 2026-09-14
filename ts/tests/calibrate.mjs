// The calibrator has to answer, cache, and never throw.
//
// Which side it lands on is the engine's business — Node is V8 and prefers
// straight-line, so that is what this asserts. `browser-tests/` is where the
// other two engines are checked.

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, {
  calibrateReroll, lastCalibration, compileTape,
  RE_ROLL_ABOVE, V8_RE_ROLL_ABOVE,
} from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
await init({ module_or_path: await readFile(resolve(here, "..", "pkg", "tapewasm_bg.wasm")) });

const fail = (m) => { console.error(`FAIL: ${m}`); process.exit(1); };

if (lastCalibration() !== null) fail("nothing measured yet, but a result is cached");

const above = await calibrateReroll();
const c = lastCalibration();
if (!c.measured) fail(`the measurement did not run: ${c.error}`);
if (above !== RE_ROLL_ABOVE && above !== V8_RE_ROLL_ABOVE) {
  fail(`${above} is neither threshold`);
}
if (above !== c.above) fail("the return value and the record disagree");
console.log(`measured: ${above} (straight ${(c.straightMs * 1000).toFixed(2)}µs, `
  + `loop ${(c.loopedMs * 1000).toFixed(2)}µs)`);

// Node is V8, and V8 prefers straight-line at the probe's size.
if (!c.prefersStraight) fail("V8 was expected to prefer straight-line here");
if (above !== V8_RE_ROLL_ABOVE) fail(`V8 should get ${V8_RE_ROLL_ABOVE}`);

// Cached: the second call is free and gives the same answer.
const t = performance.now();
const again = await calibrateReroll();
const ms = performance.now() - t;
if (again !== above) fail("a cached call gave a different answer");
if (ms > 5) fail(`a cached call took ${ms.toFixed(1)}ms, so it measured again`);

// The threshold it returns is the one `compileTape` acts on.
const tape = "n_params 2\nnew_var 0.5\nnew_var 0.5\nmul 0 1\nadd 2 0\nroot 3\n";
if (compileTape(tape, String(above)).wasm.length === 0) fail("the threshold was refused");

console.log(`cached call ${ms.toFixed(3)}ms`);
console.log("OK");
