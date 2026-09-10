// Produce what a page would serve: a module compiled ahead of time, the buffer
// sizes recorded beside it, and the draws Node gets from them.
//
// The model is written straight as tape text — the same thing a front end in
// any language hands over. Instruction numbers are what operands name, so the
// counter here is the whole bookkeeping a front end has to do.
//
// Run from browser-tests/. Needs ts/pkg/, so `make wasm` first.

import { writeFile, readFile, mkdir } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..");
const { default: init, AotSampler, compileTape, setAotExports, sharedMemory } =
  await import(resolve(repo, "ts", "index.js"));

await init({ module_or_path: await readFile(resolve(repo, "ts/pkg/tapewasm_bg.wasm")) });

const MATH = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
  tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  lgamma: () => NaN, digamma: () => NaN, phi: () => NaN,
};

const N = 40;
// A deterministic residual: with y an exact line in x the posterior for sigma
// runs off to zero, and every comparison against it divides by nothing.
let seed = 12345;
const noise = () => {
  seed = (seed * 1103515245 + 12345) & 0x7fffffff;
  return (seed / 0x7fffffff - 0.5) * 0.6;
};
const x = Array.from({ length: N }, (_, i) => -2 + i * 0.1);
const y = Array.from({ length: N }, (_, i) => -1.4 + i * 0.18 + noise());

// y ~ normal(alpha + beta * x, sigma), sampled in log sigma: normal(0,10) on
// the coefficients, exponential(1) on sigma, and the Jacobian of the transform.
const lines = [];
let next = 0;
const op = (text) => (lines.push(text), next++);

lines.push(`n_params 3`);
const alpha = op(`new_var 0.0`);
const beta = op(`new_var 0.0`);
const logSigma = op(`new_var 0.0`);
const sigma = op(`exp ${logSigma}`);
const invSigma = op(`rdiv_c ${sigma} 1.0`);
const a2 = op(`mul ${alpha} ${alpha}`);
const b2 = op(`mul ${beta} ${beta}`);
const coefPrior = op(`mul_c ${op(`add ${a2} ${b2}`)} -0.005`);
let acc = op(`sub ${coefPrior} ${sigma}`);
acc = op(`add ${acc} ${op(`mul_c ${logSigma} ${1 - N}`)}`);
for (let i = 0; i < N; i++) {
  const mu = op(`add ${alpha} ${op(`mul_c ${beta} ${x[i]}`)}`);
  const z = op(`mul ${op(`rsub_c ${mu} ${y[i]}`)} ${invSigma}`);
  acc = op(`add ${acc} ${op(`mul_c ${op(`mul ${z} ${z}`)} -0.5`)}`);
}
lines.push(`root ${acc}`);

const built = compileTape(lines.join("\n"));
const meta = {
  nParams: built.nParams,
  scratchInit: Array.from(built.scratchInit),
  paramNames: ["alpha", "beta", "log_sigma"],
  init: [0.1, 0.1, 0.1],
  warmup: 500,
  draws: 500,
  seed: 42,
  layoutId: built.layoutId,
};

const fixtures = resolve(here, "fixtures");
await mkdir(fixtures, { recursive: true });
await writeFile(resolve(fixtures, "model.wasm"), built.wasm);

// The same run, in Node, for the page to be compared against.
const aot = await WebAssembly.instantiate(built.wasm, {
  tapewasm: { memory: sharedMemory() },
  Math: MATH,
});
setAotExports(aot.instance.exports);

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
console.log(`fixtures: ${built.wasm.length} byte module, mean beta ${mean[1].toFixed(4)}`);

// `reroll` reaches the emitter. This tape is under the size threshold, so the
// default is straight-line ("never"), and "always" re-rolls it smaller.
const text = lines.join("\n");
if (compileTape(text, "never").layoutId !== built.layoutId) {
  throw new Error('compileTape(text, "never") differs from the default below the threshold');
}
const rolled = compileTape(text, "always");
if (!(rolled.wasm.length < built.wasm.length)) {
  throw new Error(`"always" is ${rolled.wasm.length} bytes against ${built.wasm.length} straight-line`);
}
let refused = false;
try { compileTape(text, "sometimes"); } catch { refused = true; }
if (!refused) throw new Error('compileTape accepted reroll "sometimes"');
const lpStraight = sampler.logProbGrad(new Float64Array(meta.init));
const rolledAot = await WebAssembly.instantiate(rolled.wasm, {
  tapewasm: { memory: sharedMemory() },
  Math: MATH,
});
setAotExports(rolledAot.instance.exports);
const lpRolled = new AotSampler(rolled.nParams, rolled.scratchInit, rolled.layoutId, [])
  .logProbGrad(new Float64Array(meta.init));
setAotExports(aot.instance.exports);
lpRolled.forEach((v, i) => {
  if (Math.abs(v - lpStraight[i]) > 1e-9 * Math.max(1, Math.abs(lpStraight[i]))) {
    throw new Error(`re-rolled [lp, grad][${i}] = ${v}, straight-line ${lpStraight[i]}`);
  }
});
console.log(`reroll: ${built.wasm.length} byte module straight-line, ${rolled.wasm.length} re-rolled`);

// sampleWithStats is sample() with each draw's statistics: the same seed draws the same.
const withStats = sampler.sampleWithStats(
  new Float64Array(meta.init), meta.warmup, meta.draws, BigInt(meta.seed), 0,
);
const total = meta.warmup + meta.draws;
const draws = withStats.draws;
if (draws.some((v, i) => v !== flat[i])) {
  throw new Error("sampleWithStats drew differently from sample() on the same seed");
}
const stats = {
  diverging: withStats.diverging, tuning: withStats.tuning, stepSize: withStats.stepSize,
  numSteps: withStats.numSteps, lp: withStats.lp,
};
for (const [name, xs] of Object.entries(stats)) {
  if (xs.length !== total) throw new Error(`${name} has ${xs.length} entries, not ${total}`);
}
if (stats.tuning.some((t, i) => t !== (i < meta.warmup ? 1 : 0))) {
  throw new Error("tuning does not mark exactly the warmup draws");
}
const lastLp = sampler.logProbGrad(draws.subarray((total - 1) * meta.nParams))[0];
if (lastLp !== stats.lp[total - 1]) {
  throw new Error(`lp ${stats.lp[total - 1]} is not the log density at the draw, ${lastLp}`);
}
console.log(
  `stats: ${stats.diverging.reduce((a, b) => a + b, 0)} divergent, ` +
  `final step ${stats.stepSize[total - 1].toPrecision(3)}`,
);

// A higher target acceptance adapts a smaller step during warmup.
const meanStep = (r) => r.stepSize.subarray(meta.warmup).reduce((a, b) => a + b, 0) / meta.draws;
const strict = new AotSampler(
  meta.nParams, new Float64Array(meta.scratchInit), meta.layoutId, meta.paramNames,
);
strict.setTargetAccept(0.95);
const tight = strict.sampleWithStats(
  new Float64Array(meta.init), meta.warmup, meta.draws, BigInt(meta.seed), 0,
);
if (!(meanStep(tight) < meanStep(withStats))) {
  throw new Error(`target 0.95 adapted step ${meanStep(tight)}, not below 0.8's ${meanStep(withStats)}`);
}
for (const bad of [0, 1, NaN]) {
  let rejected = false;
  try { strict.setTargetAccept(bad); } catch { rejected = true; }
  if (!rejected) throw new Error(`setTargetAccept(${bad}) was accepted`);
}
// The gradient-based metric estimate is a different adaptation, so different draws.
const gradBased = new AotSampler(
  meta.nParams, new Float64Array(meta.scratchInit), meta.layoutId, meta.paramNames,
);
gradBased.setGradBasedEstimate(true);
const gradDraws = gradBased.sample(
  new Float64Array(meta.init), meta.warmup, meta.draws, BigInt(meta.seed),
);
if (gradDraws.every((v, i) => v === flat[i])) {
  throw new Error("setGradBasedEstimate(true) drew exactly what the default did");
}
console.log(
  `settings: mean step ${meanStep(withStats).toPrecision(3)} at 0.8, ` +
  `${meanStep(tight).toPrecision(3)} at 0.95`,
);

// A fixture for `advi()`: two conjugate normal means, whose posterior is exactly
// Gaussian, so mean-field ADVI has a closed-form answer to recover.
const adviGroups = [
  { n: 15, sumY: 12.3 },
  { n: 25, sumY: -8.7 },
];
const priorVar = 100.0;
const coef = (n) => -(0.5 * n + 0.5 / priorVar);

const adviLines = [];
let adviNext = 0;
const adviOp = (text) => (adviLines.push(text), adviNext++);
adviLines.push(`n_params ${adviGroups.length}`);
const thetas = adviGroups.map(() => adviOp(`new_var 0.0`));
let adviAcc = null;
adviGroups.forEach(({ n, sumY }, k) => {
  const linear = adviOp(`mul_c ${thetas[k]} ${sumY}`);
  const quad = adviOp(`mul_c ${adviOp(`mul ${thetas[k]} ${thetas[k]}`)} ${coef(n)}`);
  const term = adviOp(`add ${linear} ${quad}`);
  adviAcc = adviAcc === null ? term : adviOp(`add ${adviAcc} ${term}`);
});
adviLines.push(`root ${adviAcc}`);

const adviBuilt = compileTape(adviLines.join("\n"));
const adviMeta = {
  nParams: adviBuilt.nParams,
  scratchInit: Array.from(adviBuilt.scratchInit),
  paramNames: adviGroups.map((_, k) => `theta${k}`),
  init: adviGroups.map(() => 0.0),
  numIters: 4000,
  mcSamples: 4,
  learningRate: 0.05,
  seed: 42,
  layoutId: adviBuilt.layoutId,
};

await writeFile(resolve(fixtures, "advi_model.wasm"), adviBuilt.wasm);

const adviExpected = {
  mean: adviGroups.map(({ n, sumY }) => (sumY / (1 / priorVar + n))),
  sd: adviGroups.map(({ n }) => Math.sqrt(1 / (1 / priorVar + n))),
};

await writeFile(resolve(fixtures, "advi_meta.json"), JSON.stringify(adviMeta));
await writeFile(resolve(fixtures, "advi_expected.json"), JSON.stringify(adviExpected));

// A dry run in Node, so a hyperparameter change that stops converging is
// caught here rather than only inside a browser.
const adviAot = await WebAssembly.instantiate(adviBuilt.wasm, {
  tapewasm: { memory: sharedMemory() },
  Math: MATH,
});
setAotExports(adviAot.instance.exports);
const adviSampler = new AotSampler(
  adviMeta.nParams, new Float64Array(adviMeta.scratchInit), adviMeta.layoutId, adviMeta.paramNames,
);
const adviResult = adviSampler.advi(
  new Float64Array(adviMeta.init), adviMeta.numIters, adviMeta.mcSamples,
  adviMeta.learningRate, BigInt(adviMeta.seed), 0,
);
console.log(
  `advi fixture: ${adviBuilt.wasm.length} byte module, mu ${Array.from(adviResult.mu).map((v) => v.toFixed(4))}`,
);
