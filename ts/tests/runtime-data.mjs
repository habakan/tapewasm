import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import init, {
  AotSampler,
  compileTape,
  setAotExports,
  sharedMemory,
} from "../pkg/tapewasm.js";

const here = dirname(fileURLToPath(import.meta.url));
await init({
  module_or_path: await readFile(resolve(here, "..", "pkg", "tapewasm_bg.wasm")),
});

const built = compileTape(`
n_params 1
new_var 0.0
new_data 0.0
sub 0 1
mul 2 2
mul_c 3 -0.5
root 4
outputs 4
`);
const close = (actual, expected) => {
  assert.equal(actual.length, expected.length);
  for (let i = 0; i < expected.length; i++) {
    assert.ok(Math.abs(actual[i] - expected[i]) < 1e-12, `${actual[i]} != ${expected[i]}`);
  }
};
assert.equal(built.nParams, 1);
assert.equal(built.nData, 1);

const { instance } = await WebAssembly.instantiate(built.wasm, {
  tapewasm: { memory: sharedMemory() },
  Math,
});
setAotExports(instance.exports);

const sampler = new AotSampler(
  built.nParams,
  built.scratchInit,
  built.layoutId,
  ["mu"],
);
assert.equal(sampler.nData, 1);
assert.throws(() => sampler.logProbGrad(new Float64Array([0.8])), /call setData/);

sampler.setData(new Float64Array([1.2]));
close(
  Array.from(sampler.logProbGrad(new Float64Array([0.8]))),
  [-0.08, 0.4],
);
close(Array.from(sampler.evaluate(new Float64Array([0.8]))), [-0.08]);

sampler.setData(new Float64Array([-0.2]));
close(
  Array.from(sampler.logProbGrad(new Float64Array([0.8]))),
  [-0.5, -1.0],
);

sampler.setData(new Float64Array([0.1, 0.2]));
assert.throws(() => sampler.logProbGrad(new Float64Array([0.8])), /expects 1/);

console.log("runtime data: batch values change without recompilation");
