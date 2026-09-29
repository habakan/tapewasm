function validateMetadata(metadata) {
  if (metadata === null || typeof metadata !== "object" || Array.isArray(metadata)) {
    throw new Error("metadata must be a JSON object");
  }
  if (!("nParams" in metadata)) throw new Error("metadata.nParams is required");
  if (!Number.isInteger(metadata.nParams) || metadata.nParams < 1) {
    throw new Error("metadata.nParams must be a positive integer");
  }
  for (const key of ["scratchInit", "layoutId"]) {
    if (!(key in metadata)) throw new Error(`metadata.${key} is required`);
  }
  const init = metadata.init ?? metadata.initialPoint;
  if (!Array.isArray(metadata.scratchInit) || !Array.isArray(init)) {
    throw new Error("metadata.scratchInit and metadata.init (or initialPoint) must be arrays");
  }
  if (init.length !== metadata.nParams) throw new Error("metadata.init length must match nParams");
  if (!init.every(Number.isFinite)) throw new Error("metadata.init values must be finite numbers");
  if (!metadata.scratchInit.every(Number.isFinite)) {
    throw new Error("metadata.scratchInit values must be finite numbers");
  }
  if (!Number.isInteger(metadata.layoutId)) throw new Error("metadata.layoutId must be an integer");
  if (metadata.paramNames !== undefined &&
      (!Array.isArray(metadata.paramNames) || metadata.paramNames.length !== metadata.nParams)) {
    throw new Error("metadata.paramNames length must match nParams");
  }
  return { ...metadata, init };
}

export async function loadModel({ wasmUrl, metadataUrl, workerUrl = new URL("./browser-worker.js", import.meta.url) }) {
  if (!wasmUrl || !metadataUrl) throw new TypeError("wasmUrl and metadataUrl are required");
  const response = await fetch(metadataUrl);
  if (!response.ok) throw new Error(`metadata request failed (${response.status})`);
  const metadata = validateMetadata(await response.json());
  const worker = new Worker(workerUrl, { type: "module" });
  let disposed = false;
  let busy = false;
  let nextId = 0;
  let rejectReady;
  const pending = new Map();

  worker.onmessage = ({ data }) => {
    if (data.type === "progress") {
      pending.get(data.id)?.onProgress?.(data.progress);
      return;
    }
    const task = pending.get(data.id);
    if (!task) return;
    pending.delete(data.id);
    busy = false;
    if (data.type === "error") task.reject(new Error(data.error));
    else task.resolve(data.result);
  };
  worker.onerror = (event) => {
    rejectReady?.(new Error(event.message));
    for (const task of pending.values()) task.reject(new Error(event.message));
    pending.clear();
    busy = false;
  };

  const ready = new Promise((resolve, reject) => {
    rejectReady = reject;
    const onReady = ({ data }) => {
      if (data.type === "ready") {
        worker.removeEventListener("message", onReady);
        resolve();
      } else if (data.type === "init-error") {
        worker.removeEventListener("message", onReady);
        reject(new Error(data.error));
      }
    };
    worker.addEventListener("message", onReady);
    worker.postMessage({ type: "init", wasmUrl: new URL(wasmUrl, document.baseURI).href, metadata });
  });

  try {
    await ready;
  } catch (error) {
    worker.terminate();
    throw error;
  }

  return {
    async sample({ warmup = 500, draws = 500, chains = 4, seed = 0, signal, onProgress } = {}) {
      if (disposed) throw new Error("model client is disposed");
      if (busy) throw new Error("a sampling run is already active");
      for (const [name, value] of Object.entries({ warmup, draws, chains })) {
        if (!Number.isInteger(value) || value < (name === "warmup" ? 0 : 1)) {
          throw new RangeError(`${name} must be a ${name === "warmup" ? "non-negative" : "positive"} integer`);
        }
      }
      if (signal?.aborted) throw new DOMException("The operation was aborted", "AbortError");
      if (!(typeof seed === "bigint" || (Number.isSafeInteger(seed) && seed >= 0))) {
        throw new RangeError("seed must be a non-negative safe integer or bigint");
      }
      busy = true;
      const id = ++nextId;
      return new Promise((resolve, reject) => {
        const abort = () => {
          pending.delete(id);
          busy = false;
          worker.terminate();
          disposed = true;
          reject(new DOMException("The operation was aborted", "AbortError"));
        };
        signal?.addEventListener("abort", abort, { once: true });
        pending.set(id, {
          onProgress,
          resolve: (result) => {
            signal?.removeEventListener("abort", abort);
            resolve(result);
          },
          reject: (error) => {
            signal?.removeEventListener("abort", abort);
            reject(error);
          },
        });
        worker.postMessage({ type: "sample", id, warmup, draws, chains, seed });
      });
    },
    dispose() {
      if (disposed) return;
      disposed = true;
      worker.terminate();
      for (const task of pending.values()) task.reject(new Error("model client is disposed"));
      pending.clear();
      busy = false;
    },
  };
}
