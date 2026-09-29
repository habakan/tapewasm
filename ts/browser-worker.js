import init, { AotSampler, setAotExports, sharedMemory } from "./runtime.js";

const math = {
  exp: Math.exp, log: Math.log, sin: Math.sin, cos: Math.cos, pow: Math.pow,
  tan: Math.tan, asin: Math.asin, acos: Math.acos, atan: Math.atan,
  lgamma: (x) => {
    const c = [676.5203681218851, -1259.1392167224028, 771.3234287776531,
      -176.6150291621406, 12.507343278686905, -0.13857109526572012,
      9.984369578019572e-6, 1.5056327351493116e-7];
    if (x < 0.5) return Math.log(Math.PI / Math.sin(Math.PI * x)) - math.lgamma(1 - x);
    let z = x - 1;
    let a = 0.9999999999998099;
    for (let i = 0; i < c.length; i++) a += c[i] / (z + i + 1);
    const t = z + c.length - 0.5;
    return 0.9189385332046727 + (z + 0.5) * Math.log(t) - t + Math.log(a);
  },
  digamma: (x) => {
    if (x <= 0 && Number.isInteger(x)) return NaN;
    if (x < 0.5) return math.digamma(1 - x) - Math.PI / Math.tan(Math.PI * x);
    let result = 0;
    while (x < 7) { result -= 1 / x; x += 1; }
    const inv = 1 / x;
    const inv2 = inv * inv;
    return result + Math.log(x) - inv / 2 - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 / 252));
  },
  phi: (x) => {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989422804014327 * Math.exp(-x * x / 2);
    const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return x > 0 ? 1 - p : p;
  },
};

const value = (exports, name) => exports[name] instanceof WebAssembly.Global
  ? exports[name].value : exports[name];

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "init") {
      await init();
      const response = await fetch(data.wasmUrl);
      if (!response.ok) throw new Error(`model wasm request failed (${response.status})`);
      const { instance } = await WebAssembly.instantiate(await response.arrayBuffer(), {
        tapewasm: { memory: sharedMemory() }, Math: math,
      });
      const exports = instance.exports;
      if (value(exports, "tapewasm_abi_version") !== 1) throw new Error("model Wasm ABI version is unsupported");
      if (value(exports, "tapewasm_layout_id") !== data.metadata.layoutId) {
        throw new Error("metadata.layoutId does not match model Wasm");
      }
      setAotExports(exports);
      self.model = { metadata: data.metadata, exports };
      self.postMessage({ type: "ready" });
      return;
    }
    if (data.type === "sample") {
      const { metadata, exports } = self.model;
      const sampler = new AotSampler(metadata.nParams, new Float64Array(metadata.scratchInit),
        metadata.layoutId, metadata.paramNames ?? []);
      const chains = [];
      const diverging = [];
      try {
        for (let chain = 0; chain < data.chains; chain++) {
          self.postMessage({ type: "progress", id: data.id, progress: { state: "sampling", chain } });
          const result = sampler.sampleWithStats(new Float64Array(metadata.init), data.warmup,
            data.draws, BigInt(data.seed) + BigInt(chain), chain);
          try {
            chains.push(result.draws.slice(data.warmup * metadata.nParams));
            diverging.push(result.diverging.slice(data.warmup));
          } finally {
            result.free();
          }
        }
      } finally {
        sampler.free();
        setAotExports(exports);
      }
      const buffers = [...chains, ...diverging].map((array) => array.buffer);
      self.postMessage({ type: "result", id: data.id, result: {
        parameterNames: metadata.paramNames ?? [], space: "as-compiled", nParams: metadata.nParams,
        nDraws: data.draws, chains, diverging,
      } }, buffers);
    }
  } catch (error) {
    self.postMessage({ type: data.type === "init" ? "init-error" : "error",
      id: data.id, error: error instanceof Error ? error.message : String(error) });
  }
};
