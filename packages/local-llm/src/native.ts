import { createRequire } from 'node:module';

declare const __brand: unique symbol;
type Brand<T, B> = T & { readonly [__brand]: B };

export type NativeModel = Brand<object, 'NativeModel'>;
export type NativeContext = Brand<object, 'NativeContext'>;
export type NativeMtmdContext = Brand<object, 'NativeMtmdContext'>;

export interface NativeAddon {
  backendInfo(): string;
  backendVersion(): string;
  apiVersion(): number;

  loadModel(
    path: string,
    options?: { n_gpu_layers?: number; use_mmap?: boolean },
  ): NativeModel;

  createContext(
    model: NativeModel,
    options?: {
      n_ctx?: number;
      n_batch?: number;
      n_threads?: number;
      flash_attn_type?: number;
      type_k?: number;
      type_v?: number;
      n_seq_max?: number;
      draft_model?: NativeModel;
      draft_n_max?: number;
    },
  ): NativeContext;

  tokenize(
    model: NativeModel,
    text: string,
    addSpecial?: boolean,
    parseSpecial?: boolean,
  ): Int32Array;

  detokenize(model: NativeModel, tokens: Int32Array | number[]): string;

  applyChatTemplate(
    model: NativeModel,
    messages: { role: string; content: string }[],
    addAssistant?: boolean,
  ): string;

  inferSync(
    model: NativeModel,
    ctx: NativeContext,
    prompt: string,
    options?: {
      max_tokens?: number;
      temperature?: number;
      top_p?: number;
      top_k?: number;
      repeat_penalty?: number;
      frequency_penalty?: number;
      presence_penalty?: number;
      seed?: number;
      grammar?: string;
      grammar_root?: string;
    },
  ): string;

  inferStream(
    model: NativeModel,
    ctx: NativeContext,
    prompt: string,
    options: {
      max_tokens?: number;
      temperature?: number;
      top_p?: number;
      top_k?: number;
      repeat_penalty?: number;
      frequency_penalty?: number;
      presence_penalty?: number;
      seed?: number;
      n_past?: number;
      grammar?: string;
      grammar_root?: string;
    },
    callback: (error: Error | null, token: string | null) => void,
  ): void;

  loadModelFromBuffer(
    buffer: Buffer,
    options?: { n_gpu_layers?: number; use_mmap?: boolean },
  ): NativeModel;

  getModelSize(model: NativeModel): number;

  freeModel(model: NativeModel): void;
  freeContext(ctx: NativeContext): void;

  /** Returns the actual context window size (in tokens) for this context. */
  getContextSize(ctx: NativeContext): number;

  /** Run a single-token warmup pass to prime GPU/CPU caches. */
  warmup(model: NativeModel, ctx: NativeContext): void;

  kvCacheClear(ctx: NativeContext, fromPos: number): void;
  stopStream(ctx: NativeContext): void;

  /** Returns true if the OS is in low-power / battery-saver mode. */
  isLowPowerMode?(): boolean;

  getPerf(ctx: NativeContext): {
    promptEvalMs: number;
    generationMs: number;
    promptTokens: number;
    generatedTokens: number;
    promptTokensPerSec: number;
    generatedTokensPerSec: number;
  } | null;

  createMtmdContext(
    model: NativeModel,
    projectorPath: string,
    options?: { use_gpu?: boolean; n_threads?: number },
  ): NativeMtmdContext;

  freeMtmdContext(ctx: NativeMtmdContext): void;

  supportVision(ctx: NativeMtmdContext): boolean;

  inferSyncVision(
    model: NativeModel,
    ctx: NativeContext,
    mtmdCtx: NativeMtmdContext,
    prompt: string,
    imageBuffers: Buffer[],
    options?: {
      max_tokens?: number;
      temperature?: number;
      top_p?: number;
      top_k?: number;
      repeat_penalty?: number;
      frequency_penalty?: number;
      presence_penalty?: number;
      seed?: number;
      grammar?: string;
      grammar_root?: string;
    },
  ): string;

  inferStreamVision(
    model: NativeModel,
    ctx: NativeContext,
    mtmdCtx: NativeMtmdContext,
    prompt: string,
    imageBuffers: Buffer[],
    options: {
      max_tokens?: number;
      temperature?: number;
      top_p?: number;
      top_k?: number;
      repeat_penalty?: number;
      frequency_penalty?: number;
      presence_penalty?: number;
      seed?: number;
      grammar?: string;
      grammar_root?: string;
    },
    callback: (error: Error | null, token: string | null) => void,
  ): void;

  inferBatch(
    model: NativeModel,
    ctx: NativeContext,
    prompts: string[],
    options: Array<{
      max_tokens?: number;
      temperature?: number;
      top_p?: number;
      top_k?: number;
      repeat_penalty?: number;
      frequency_penalty?: number;
      presence_penalty?: number;
      seed?: number;
      grammar?: string;
      grammar_root?: string;
    }>,
    callback: (error: Error | null, token: string | null, seqIndex: number, finishReason: string | null) => void,
  ): void;

  /** Convert a JSON Schema (as a JSON string) to a GBNF grammar string. */
  jsonSchemaToGrammar(schemaJson: string): string;

  /** Returns the embedding dimension for the model. */
  getEmbeddingDimension(model: NativeModel): number;

  /** Creates a context with embeddings enabled and the specified pooling type. */
  createEmbeddingContext(
    model: NativeModel,
    options?: { n_ctx?: number; n_batch?: number; n_threads?: number; pooling_type?: number },
  ): NativeContext;

  /** Encode tokens and return the L2-normalized embedding vector. */
  embed(ctx: NativeContext, model: NativeModel, tokens: Int32Array): Float32Array;

  /** Encode multiple token arrays in a single batch and return an array of L2-normalized embeddings. */
  embedBatch(ctx: NativeContext, model: NativeModel, tokenArrays: Int32Array[]): Float32Array[];

  quantize(
    inputPath: string,
    outputPath: string,
    options: {
      ftype: number;
      nthread?: number;
      allow_requantize?: boolean;
      quantize_output_tensor?: boolean;
      pure?: boolean;
    },
    callback: (error: Error | null) => void,
  ): void;

  setLogCallback(
    callback: ((level: number, text: string) => void) | null,
  ): void;
  setLogLevel(level: number): void;

  /** Returns the C-engine-computed optimal thread count for this machine. */
  optimalThreadCount(): number;

  /** Run a benchmark entirely in the C engine. Returns averaged metrics. */
  benchmark(
    model: NativeModel,
    ctx: NativeContext,
    options?: { promptTokens?: number; generateTokens?: number; iterations?: number },
  ): {
    promptTokensPerSec: number;
    generatedTokensPerSec: number;
    ttftMs: number;
    totalMs: number;
    iterations: number;
  };
}

let cached: NativeAddon | null = null;

/** Allow React Native (or other runtimes) to inject their own native backend. */
export function setNativeAddon(addon: NativeAddon): void {
  cached = addon;
}

export function loadNativeAddon(): NativeAddon {
  if (cached) return cached;
  const require = createRequire(import.meta.url);

  // 1. Try platform-specific prebuilt binary
  const platformPkg = `@local-llm/${process.platform}-${process.arch}`;
  try {
    cached = require(platformPkg) as NativeAddon;
    return cached;
  } catch {
    // Not installed — fall through
  }

  // 2. Fall back to workspace native package (local dev)
  try {
    cached = require('@local-llm/native') as NativeAddon;
    return cached;
  } catch {
    // Not available either
  }

  throw new Error(
    `No native binary found for ${process.platform}-${process.arch}.\n` +
    `Install the platform package: npm install ${platformPkg}\n` +
    `Or build from source: cd packages/native && npm run build`,
  );
}
