// Public facade: re-exports the wasm-bindgen bindings so callers never reach
// into `pkg/`. Plain `.js` because a package entry point has to load without
// `--experimental-strip-types`, and there is no TypeScript syntax here to lose.

import init from "./pkg/tapewasm.js";
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
export {
  calibrateReroll,
  compileTapeCalibrated,
  lastCalibration,
  RE_ROLL_ABOVE,
  V8_RE_ROLL_ABOVE,
} from "./calibrate.js";
export default init;
