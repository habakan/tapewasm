export interface SamplingProgress {
  state: "sampling";
  chain: number;
}

export interface SampleOptions {
  warmup?: number;
  draws?: number;
  chains?: number;
  seed?: number | bigint;
  signal?: AbortSignal;
  onProgress?: (progress: SamplingProgress) => void;
}

export interface Fit {
  parameterNames: string[];
  space: "as-compiled";
  nParams: number;
  nDraws: number;
  chains: Float64Array[];
  diverging: Uint8Array[];
}

export interface ModelClient {
  sample(options?: SampleOptions): Promise<Fit>;
  dispose(): void;
}

export function loadModel(options: {
  wasmUrl: string | URL;
  metadataUrl: string | URL;
  workerUrl?: string | URL;
}): Promise<ModelClient>;
