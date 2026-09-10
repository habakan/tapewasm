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
