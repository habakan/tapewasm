// Produce what a page would serve: a module compiled ahead of time, the buffer
// sizes recorded beside it, and the draws Node gets from them.
//
// Run from browser-tests/. Needs ts/pkg/, so `make wasm` first.

import { writeFile, readFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..");
const { default: init, StanModel, AotSampler, setAotExports, sharedMemory } =
  await import(resolve(repo, "ts", "index.js"));

await init({ module_or_path: await readFile(resolve(repo, "ts/pkg/stanwasm_bg.wasm")) });

const N = 40;
// A deterministic residual: with y an exact line in x the posterior for sigma
// runs off to zero, and every comparison against it divides by nothing.
let seed = 12345;
const noise = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return (seed / 0x7fffffff - 0.5) * 0.6;
};
const model = new StanModel(
  `data { int<lower=0> N; vector[N] x; vector[N] y; }
   parameters { real alpha; real beta; real<lower=0> sigma; }
   model {
     alpha ~ normal(0, 10); beta ~ normal(0, 10); sigma ~ exponential(1);
     y ~ normal(alpha + beta * x, sigma);
   }`,
  JSON.stringify({
    N,
    x: Array.from({ length: N }, (_, i) => -2 + i * 0.1),
    y: Array.from({ length: N }, (_, i) => -1.4 + i * 0.18 + noise()),
  }),
);

const moduleBytes = model.compileToWasm();
const meta = {
  nParams: model.n_params,
  scratchInit: Array.from(model.aotScratchInit()),
  paramNames: model.paramNames(),
  init: [0, 0, 0],
  warmup: 500,
  draws: 500,
  seed: 42,
};

const fixtures = resolve(here, "fixtures");
await mkdir(fixtures, { recursive: true });
await writeFile(resolve(fixtures, "model.wasm"), moduleBytes);

// The same run, in Node, for the page to be compared against.
const imports = {
  stan: { memory: sharedMemory() },
  Math: {
    exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
    tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
    lgamma: () => NaN, digamma: () => NaN, phi: () => NaN,
  },
};
const aot = await WebAssembly.instantiate(moduleBytes, imports);
setAotExports(aot.instance.exports);
meta.layoutId = aot.instance.exports.stanwasm_layout_id.value >>> 0;

const sampler = new AotSampler(
  meta.nParams, new Float64Array(meta.scratchInit), meta.layoutId, meta.paramNames,
);
const flat = sampler.sample(
  new Float64Array(meta.init), meta.warmup, meta.draws, BigInt(meta.seed),
);
const post = flat.subarray(meta.warmup * meta.nParams);
const mean = new Array(meta.nParams).fill(0);
const sd = new Array(meta.nParams).fill(0);
for (let i = 0; i < meta.draws; i++) {
  for (let k = 0; k < meta.nParams; k++) mean[k] += post[i * meta.nParams + k] / meta.draws;
}
for (let i = 0; i < meta.draws; i++) {
  for (let k = 0; k < meta.nParams; k++) {
    sd[k] += (post[i * meta.nParams + k] - mean[k]) ** 2 / meta.draws;
  }
}

await writeFile(resolve(fixtures, "meta.json"), JSON.stringify(meta));
await writeFile(
  resolve(fixtures, "expected.json"),
  JSON.stringify({ mean, sd: sd.map(Math.sqrt) }),
);
console.log(`fixtures: ${moduleBytes.length} byte module, mean beta ${mean[1].toFixed(4)}`);
