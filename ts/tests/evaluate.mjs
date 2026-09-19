// `evaluate` through the package: a tape that names outputs reports them at a
// point, and one that names none refuses.
//
// The model is two observations of y ~ normal(mu, sigma) with sigma fixed, so
// each term is checkable in a line of JavaScript.

import { readFile } from "node:fs/promises";
import { resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import init, {
  AotSampler, compileTape, setAotExports, sharedMemory,
} from "../index.js";

const here = dirname(fileURLToPath(import.meta.url));
await init({ module_or_path: await readFile(resolve(here, "..", "pkg", "tapewasm_bg.wasm")) });

const fail = (m) => { console.error(`FAIL: ${m}`); process.exit(1); };

const y = [1.4, -0.3];
const SIGMA = 0.5;
const HALF_LOG_2PI = 0.5 * Math.log(2 * Math.PI);

// One parameter, mu. Each observation's own log-likelihood term is a node, and
// the density is their sum plus a normal(0, 10) prior on mu.
const lines = [];
let next = 0;
const op = (text) => (lines.push(text), next++);
lines.push("n_params 1");
const mu = op("new_var 0.0");
const terms = y.map((yi) => {
  const z = op(`mul_c ${op(`rsub_c ${mu} ${yi}`)} ${1 / SIGMA}`);
  const sq = op(`mul_c ${op(`mul ${z} ${z}`)} -0.5`);
  return op(`add_c ${sq} ${-Math.log(SIGMA) - HALF_LOG_2PI}`);
});
const prior = op(`mul_c ${op(`mul ${mu} ${mu}`)} -0.005`);
const root = terms.reduce((acc, t) => op(`add ${acc} ${t}`), prior);
lines.push(`root ${root}`);
lines.push(`outputs ${terms.join(" ")}`);

const built = compileTape(lines.join("\n"));
if (built.nOutputs !== y.length) fail(`nOutputs ${built.nOutputs}, not ${y.length}`);

const instantiate = async (b) => {
  const aot = await WebAssembly.instantiate(b.wasm, {
    tapewasm: { memory: sharedMemory() },
    Math,
  });
  setAotExports(aot.instance.exports);
  return new AotSampler(b.nParams, b.scratchInit, b.layoutId, ["mu"]);
};

const sampler = await instantiate(built);
const at = 0.7;
const got = sampler.evaluate(new Float64Array([at]));
const want = y.map((yi) =>
  -0.5 * ((yi - at) / SIGMA) ** 2 - Math.log(SIGMA) - HALF_LOG_2PI);
if (got.length !== want.length) fail(`evaluate returned ${got.length} values`);
got.forEach((v, i) => {
  if (Math.abs(v - want[i]) > 1e-12) fail(`term ${i} = ${v}, expected ${want[i]}`);
});

// The two entry points share one scratch buffer, so interleaving them has to
// leave both right — and the terms have to add up to the density they came from.
const lp = sampler.logProbGrad(new Float64Array([at]));
const again = sampler.evaluate(new Float64Array([at]));
if (again.some((v, i) => v !== got[i])) fail("evaluate answered differently after logProbGrad");
const sum = got.reduce((a, b) => a + b, 0) + -0.005 * at * at;
if (Math.abs(sum - lp[0]) > 1e-12) fail(`terms sum to ${sum}, density is ${lp[0]}`);

// A draw at a time is the shape a pointwise log-likelihood is assembled in.
const draws = sampler.sample(new Float64Array([0.1]), 200, 200, 42n).subarray(200);
const pointwise = Array.from({ length: 200 }, (_, i) => sampler.evaluate(draws.subarray(i, i + 1)));
if (pointwise.some((row) => row.length !== y.length || row.some((v) => !Number.isFinite(v)))) {
  fail("a draw's log-likelihood row is not finite");
}

// Nothing named, nothing to report — and the refusal says what to do about it.
lines.pop();
const bare = compileTape(lines.join("\n"));
if (bare.nOutputs !== 0) fail("a tape without an outputs line reported outputs");
const plain = await instantiate(bare);
let refused = "";
try { plain.evaluate(new Float64Array([at])); } catch (e) { refused = String(e.message ?? e); }
if (!refused.includes("outputs")) fail(`the refusal does not mention outputs: ${refused}`);

console.log(`evaluate: ${y.length} terms, summing to the density at mu=${at}`);
