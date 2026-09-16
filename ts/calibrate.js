// Which shape this engine prefers, measured once.
//
// `compileTape`'s `"auto"` threshold cannot be right for every engine:
// straight-line and re-rolled cross over around 60,000 nodes in V8 and around
// 2,000 in SpiderMonkey and JavaScriptCore. The built-in value serves the
// latter two and costs V8 up to 7.6x on a trace between them. This measures
// which side the engine is on and returns a threshold to pass to `compileTape`.
//
//   await init(...);
//   const above = await calibrateReroll();
//   const built = compileTape(text, String(above));
//
// The answer is cached: it is a property of the engine, not of the model.

import { compileTape } from "./pkg/tapewasm.js";

/// What `"auto"` uses. Right for SpiderMonkey and JavaScriptCore.
export const RE_ROLL_ABOVE = 2_000;

/// For an engine that prefers straight-line, as V8 does.
///
/// Bracketed by measurement rather than picked: on eleven posteriordb models,
/// straight-line still wins at 8,026 nodes and has lost by 24,564. A threshold
/// of 24,000 already costs `low_dim_gauss_mix` 1.21x, so this sits clear of
/// that edge. At 20,000 the five models whose shape changes get 2.54x in the
/// geometric mean and none is slower.
export const V8_RE_ROLL_ABOVE = 20_000;

/// Where the two engine families disagree most clearly. At this size V8 prefers
/// straight-line on every model measured, and the other two prefer loops on all
/// but one.
const PROBE_NODES = 4_000;

let cached = null;

/// A tape of the commonest shape — one accumulated term per observation —
/// using nothing but arithmetic, so the module it compiles to imports only
/// memory and the probe needs no maths from the caller.
function probeTape(terms) {
  // Instruction 0, 1, 2 are the parameters; `at` is the index the next line takes.
  const out = ["n_params 3", "new_var 0.4", "new_var 1.1", "new_var 0.3"];
  let at = 3;
  const push = (line) => {
    out.push(line);
    return at++;
  };
  let acc = null;
  for (let i = 0; i < terms; i++) {
    const x = (0.001 * (i % 997)).toFixed(6);
    const y = (0.5 + 0.001 * (i % 991)).toFixed(6);
    const bx = push(`mul_c 1 ${x}`);     // beta * x
    const mu = push(`add ${bx} 0`);      // + alpha
    const r = push(`rsub_c ${mu} ${y}`); // y - mu
    const z = push(`div ${r} 2`);        // / sigma
    const sq = push(`mul ${z} ${z}`);
    acc = acc === null ? sq : push(`add ${acc} ${sq}`);
  }
  const half = push(`mul_c ${acc} -0.5`);
  out.push(`root ${half}`);
  return out.join("\n") + "\n";
}

async function timeOne(wasm, nParams, scratchInit, rounds) {
  const need = nParams * 16 + scratchInit.length * 8;
  const memory = new WebAssembly.Memory({ initial: Math.ceil(need / 65536) + 2 });
  const { instance } = await WebAssembly.instantiate(wasm, { tapewasm: { memory } });
  const view = new Float64Array(memory.buffer);
  view.set(scratchInit, nParams * 2);
  for (let i = 0; i < nParams; i++) view[i] = 0.1 * (i + 1);
  const lpg = instance.exports.log_prob_grad;
  const call = () => lpg(0, nParams * 8, nParams, nParams * 16);

  for (let i = 0; i < 50; i++) call();          // past the first tier
  const t0 = performance.now();
  let n = 0;
  while (performance.now() - t0 < 4) { call(); n++; }
  const iters = Math.max(1, n);
  let best = Infinity;
  for (let r = 0; r < rounds; r++) {
    const t = performance.now();
    for (let j = 0; j < iters; j++) call();
    best = Math.min(best, (performance.now() - t) / iters);
  }
  return best;
}

/**
 * Measure which shape this engine prefers and return a threshold for
 * `compileTape`'s `reroll` argument. Call after `init()`.
 *
 * Cached after the first call — pass `{ force: true }` to measure again.
 * On any failure it returns the built-in threshold, so a caller can use the
 * result without guarding it.
 */
export async function calibrateReroll({ force = false, rounds = 7 } = {}) {
  if (cached !== null && !force) return cached.above;
  let result;
  try {
    const text = probeTape(Math.max(1, Math.round(PROBE_NODES / 5)));
    const straight = compileTape(text, "never");
    const looped = compileTape(text, "always");
    // Interleaved, so a drift in machine state lands on both.
    let sBest = Infinity, lBest = Infinity;
    for (let r = 0; r < 2; r++) {
      sBest = Math.min(sBest, await timeOne(straight.wasm, straight.nParams, straight.scratchInit, rounds));
      lBest = Math.min(lBest, await timeOne(looped.wasm, looped.nParams, looped.scratchInit, rounds));
    }
    const prefersStraight = sBest < lBest;
    result = {
      above: prefersStraight ? V8_RE_ROLL_ABOVE : RE_ROLL_ABOVE,
      prefersStraight,
      straightMs: sBest,
      loopedMs: lBest,
      measured: true,
    };
  } catch (e) {
    // A threshold is an optimisation; failing to measure one is not an error.
    result = { above: RE_ROLL_ABOVE, measured: false, error: String(e?.message ?? e) };
  }
  cached = result;
  return result.above;
}

/** What the last `calibrateReroll` measured, or `null` before the first call. */
export function lastCalibration() {
  return cached;
}
