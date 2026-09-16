// Types for `index.js`; the shapes come from `pkg/`, which `make wasm` writes.
export {
  AotSampler,
  AdviResult,
  CompiledTape,
  SampleResult,
  compileTape,
  tapewasmVersion,
  setAotExports,
  clearAotExports,
  sharedMemory,
} from "./pkg/tapewasm.js";
export { default } from "./pkg/tapewasm.js";

/** What `"auto"` uses: right for SpiderMonkey and JavaScriptCore. */
export const RE_ROLL_ABOVE: number;
/** For an engine that prefers straight-line, as V8 does. */
export const V8_RE_ROLL_ABOVE: number;

export interface Calibration {
  /** Threshold to pass to `compileTape`. */
  above: number;
  /** False when the measurement could not run; `above` is the built-in value. */
  measured: boolean;
  prefersStraight?: boolean;
  straightMs?: number;
  loopedMs?: number;
  error?: string;
}

/**
 * Measure which shape this engine prefers and return a threshold for
 * `compileTape`'s `reroll` argument. Call after `init()`. Cached; pass
 * `{ force: true }` to measure again. Never throws — on failure it returns
 * the built-in threshold.
 */
export function calibrateReroll(options?: { force?: boolean; rounds?: number }): Promise<number>;

/** What the last `calibrateReroll` measured, or `null` before the first call. */
export function lastCalibration(): Calibration | null;
