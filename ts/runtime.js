import init from "./pkg-sampler/tapewasm.js";

export {
  AotSampler,
  SampleResult,
  clearAotExports,
  setAotExports,
  sharedMemory,
  tapewasmVersion,
} from "./pkg-sampler/tapewasm.js";

export default function initSampler(input = new URL("./pkg-sampler/tapewasm_bg.wasm", import.meta.url)) {
  return init(input);
}
