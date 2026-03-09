import { readFile } from 'node:fs/promises';
import { cpus } from 'node:os';
import { loadNativeAddon, type NativeAddon, type NativeModel, type NativeContext, type NativeMtmdContext } from './native.js';
import type { ModelOptions, ContextOptions, EmbeddingContextOptions, EmbeddingPoolingType, GenerateOptions, ChatMessage, ContentPart, ComputeMode, ResponseFormat, FlashAttentionMode, KvCacheType, QuantizeOptions, QuantizationType, BatchResult, InferenceMetrics, BenchmarkOptions, BenchmarkResult } from './types.js';

/** Strip keys whose value is undefined so the native addon doesn't see them. */
function defined<T extends Record<string, unknown>>(obj: T): Partial<T> {
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(obj)) {
    if (v !== undefined) out[k] = v;
  }
  return out as Partial<T>;
}

/**
 * Simple async mutex — serializes access to a shared resource.
 * Callers do `const release = await mutex.acquire()` and call `release()` when done.
 */
class Mutex {
  private _queue: Array<() => void> = [];
  private _locked = false;

  acquire(): Promise<() => void> {
    return new Promise<() => void>((resolve) => {
      const tryAcquire = () => {
        this._locked = true;
        resolve(() => {
          this._locked = false;
          const next = this._queue.shift();
          if (next) next();
        });
      };

      if (!this._locked) {
        tryAcquire();
      } else {
        this._queue.push(tryAcquire);
      }
    });
  }
}

/**
 * Detect the optimal thread count for LLM inference.
 * Delegates to the C engine which uses platform-specific APIs:
 * - macOS: P-core count via sysctl, low-power mode detection
 * - Linux: big.LITTLE detection via cpufreq sysfs, battery state
 * - Fallback: half of hardware_concurrency (avoids hyperthreads)
 */
export function optimalThreadCount(): number {
  try {
    const addon = loadNativeAddon();
    return addon.optimalThreadCount();
  } catch {
    // Native addon not loaded yet — JS fallback
    const cores = cpus();
    if (cores.length === 0) return 4;
    return Math.max(1, Math.floor(cores.length / 2));
  }
}

export class Model {
  private native: NativeAddon;
  private handle: NativeModel | null;
  private mtmdHandle: NativeMtmdContext | null = null;

  constructor(path: string, options?: ModelOptions) {
    this.native = loadNativeAddon();
    const gpuLayers = options?.gpuLayers ?? resolveCompute(options?.compute, this.native);
    try {
      this.handle = this.native.loadModel(path, defined({
        n_gpu_layers: gpuLayers,
        use_mmap: options?.useMmap,
      }));
    } catch (err) {
      throw wrapNativeError(err, 'model-load', path);
    }
  }

  /** Async factory — works on both Node.js and React Native. */
  static async load(path: string, options?: ModelOptions): Promise<Model> {
    const instance = Object.create(Model.prototype) as Model;
    instance.native = loadNativeAddon();
    const gpuLayers = options?.gpuLayers ?? resolveCompute(options?.compute, instance.native);
    try {
      const result = instance.native.loadModel(path, defined({
        n_gpu_layers: gpuLayers,
        use_mmap: options?.useMmap,
      }));
      instance.handle = result instanceof Promise ? await result : result;
    } catch (err) {
      throw wrapNativeError(err, 'model-load', path);
    }
    instance.mtmdHandle = null;
    return instance;
  }

  static fromBuffer(buffer: Buffer, options?: ModelOptions): Model {
    const instance = Object.create(Model.prototype) as Model;
    instance.native = loadNativeAddon();
    const gpuLayers = options?.gpuLayers ?? resolveCompute(options?.compute, instance.native);
    try {
      instance.handle = instance.native.loadModelFromBuffer(buffer, defined({
        n_gpu_layers: gpuLayers,
        use_mmap: false,
      }));
    } catch (err) {
      throw wrapNativeError(err, 'model-load', '<buffer>');
    }
    return instance;
  }

  get sizeBytes(): number {
    return this.native.getModelSize(this.ensureHandle());
  }

  /** The underlying native model handle (for passing to createContext as draft model). */
  get nativeHandle(): NativeModel {
    return this.ensureHandle();
  }

  private ensureHandle(): NativeModel {
    if (!this.handle) throw new Error('Model has been disposed');
    return this.handle;
  }

  loadProjector(projectorPath: string, options?: { useGpu?: boolean; threads?: number }): void {
    try {
      this.mtmdHandle = this.native.createMtmdContext(this.ensureHandle(), projectorPath, {
        use_gpu: options?.useGpu,
        n_threads: options?.threads,
      });
    } catch (err) {
      throw wrapNativeError(err, 'projector', projectorPath);
    }
  }

  get isMultimodal(): boolean {
    return this.mtmdHandle != null;
  }

  get mtmdContext(): NativeMtmdContext | null {
    return this.mtmdHandle;
  }

  createContext(options?: ContextOptions & { draftModel?: NativeModel }): InferenceContext {
    try {
      const handle = this.native.createContext(this.ensureHandle(), defined({
        n_ctx: options?.contextSize,
        n_batch: options?.batchSize,
        n_threads: options?.threads ?? optimalThreadCount(),
        flash_attn_type: options?.flashAttention != null ? flashAttnToNative(options.flashAttention) : undefined,
        type_k: options?.kvCacheTypeK ? kvCacheTypeToNative(options.kvCacheTypeK) : undefined,
        type_v: options?.kvCacheTypeV ? kvCacheTypeToNative(options.kvCacheTypeV) : undefined,
        n_seq_max: options?.maxSequences,
        draft_model: options?.draftModel,
        draft_n_max: options?.draftNMax,
      }));
      return new InferenceContext(this.native, this.ensureHandle(), handle, this.mtmdHandle);
    } catch (err) {
      throw wrapNativeError(err, 'context-create');
    }
  }

  createEmbeddingContext(options?: EmbeddingContextOptions): EmbeddingContext {
    try {
      const handle = this.native.createEmbeddingContext(this.ensureHandle(), defined({
        n_ctx: options?.contextSize,
        n_batch: options?.batchSize,
        n_threads: options?.threads ?? optimalThreadCount(),
        pooling_type: options?.poolingType ? poolingTypeToNative(options.poolingType) : undefined,
      }));
      const dimension = this.native.getEmbeddingDimension(this.ensureHandle());
      return new EmbeddingContext(this.native, this.ensureHandle(), handle, dimension);
    } catch (err) {
      throw wrapNativeError(err, 'context-create');
    }
  }

  /** Returns the embedding dimension for this model. */
  get embeddingDimension(): number {
    return this.native.getEmbeddingDimension(this.ensureHandle());
  }

  tokenize(text: string): Int32Array {
    return this.native.tokenize(this.ensureHandle(), text);
  }

  detokenize(tokens: Int32Array | number[]): string {
    return this.native.detokenize(this.ensureHandle(), tokens);
  }

  applyChatTemplate(messages: ChatMessage[], addAssistant?: boolean): string {
    const plain = messages.map((m) => ({
      role: m.role,
      content: typeof m.content === 'string'
        ? m.content
        : m.content.filter((p) => p.type === 'text').map((p) => p.type === 'text' ? p.text : '').join(''),
    }));
    return this.native.applyChatTemplate(this.ensureHandle(), plain, addAssistant);
  }

  dispose(): void {
    if (this.mtmdHandle) {
      this.native.freeMtmdContext(this.mtmdHandle);
      this.mtmdHandle = null;
    }
    if (this.handle) {
      this.native.freeModel(this.handle);
      this.handle = null;
    }
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

export class InferenceContext {
  private native: NativeAddon;
  private model: NativeModel;
  private handle: NativeContext | null;
  private mtmdCtx: NativeMtmdContext | null;
  private mutex = new Mutex();
  /** Token IDs currently in the KV cache (prompt + generated from last call). */
  private cachedTokenIds: number[] = [];

  /** @internal — use Model.createContext() instead */
  constructor(native: NativeAddon, model: NativeModel, handle: NativeContext, mtmdCtx?: NativeMtmdContext | null) {
    this.native = native;
    this.model = model;
    this.handle = handle;
    this.mtmdCtx = mtmdCtx ?? null;
  }

  /** The actual context window size in tokens (as allocated by llama.cpp). */
  get contextSize(): number {
    return this.native.getContextSize(this.ensureHandle());
  }

  private ensureHandle(): NativeContext {
    if (!this.handle) throw new Error('Context has been disposed');
    return this.handle;
  }

  /**
   * Compares the new prompt tokens against the cached KV state,
   * clears divergent cache entries, and returns the number of
   * tokens that can be skipped (already in KV cache).
   */
  private prepareKvCache(prompt: string): { promptTokens: number[]; nPast: number } {
    const promptTokens = Array.from(this.native.tokenize(this.model, prompt));
    const commonLen = findCommonPrefixLength(this.cachedTokenIds, promptTokens);

    const ctx = this.ensureHandle();
    if (commonLen < this.cachedTokenIds.length) {
      this.native.kvCacheClear(ctx, commonLen);
    }

    return { promptTokens, nPast: commonLen };
  }

  /**
   * Updates the cached token IDs after a generation completes.
   * Includes both the prompt tokens and the generated tokens.
   */
  private updateKvCache(promptTokens: number[], generatedText: string): void {
    if (generatedText) {
      const genTokens = Array.from(this.native.tokenize(this.model, generatedText, false));
      this.cachedTokenIds = [...promptTokens, ...genTokens];
    } else {
      this.cachedTokenIds = promptTokens;
    }
  }

  async generate(prompt: string, options?: GenerateOptions): Promise<string> {
    if (options?.signal?.aborted) {
      throw new Error('Generation aborted');
    }
    const release = await this.mutex.acquire();
    try {
      const ctx = this.ensureHandle();
      const { promptTokens, nPast } = this.prepareKvCache(prompt);
      const opts = { ...mapGenerateOptions(this.native, options), n_past: nPast };
      let text = await this.collectStream(
        (cb) => this.native.inferStream(this.model, ctx, prompt, opts, cb),
        options?.signal,
      );
      if (options?.stop?.length) {
        text = truncateAtStopSequence(text, options.stop);
      }
      this.updateKvCache(promptTokens, text);
      return text;
    } catch (err) {
      throw wrapNativeError(err, 'inference');
    } finally {
      release();
    }
  }

  async *stream(prompt: string, options?: GenerateOptions): AsyncGenerator<string> {
    if (options?.signal?.aborted) throw new Error('Generation aborted');

    const release = await this.mutex.acquire();
    const ctx = this.ensureHandle();
    const { promptTokens, nPast } = this.prepareKvCache(prompt);
    const opts = { ...mapGenerateOptions(this.native, options), n_past: nPast };

    try {
      const generatedParts: string[] = [];
      for await (const token of this.streamInternal(
        (cb) => this.native.inferStream(this.model, ctx, prompt, opts, cb),
        options,
      )) {
        generatedParts.push(token);
        yield token;
      }
      this.updateKvCache(promptTokens, generatedParts.join(''));
    } finally {
      release();
    }
  }

  async generateVision(prompt: string, imageBuffers: Buffer[], options?: GenerateOptions): Promise<string> {
    if (options?.signal?.aborted) {
      throw new Error('Generation aborted');
    }
    if (!this.mtmdCtx) throw new Error('No projector loaded — call loadProjector() first');
    const release = await this.mutex.acquire();
    this.cachedTokenIds = [];
    try {
      const ctx = this.ensureHandle();
      const mtmd = this.mtmdCtx;
      const opts = mapGenerateOptions(this.native, options);
      let text = await this.collectStream(
        (cb) => this.native.inferStreamVision(this.model, ctx, mtmd, prompt, imageBuffers, opts, cb),
        options?.signal,
      );
      if (options?.stop?.length) {
        text = truncateAtStopSequence(text, options.stop);
      }
      return text;
    } catch (err) {
      throw wrapNativeError(err, 'inference');
    } finally {
      release();
    }
  }

  async *streamVision(prompt: string, imageBuffers: Buffer[], options?: GenerateOptions): AsyncGenerator<string> {
    if (options?.signal?.aborted) throw new Error('Generation aborted');
    if (!this.mtmdCtx) throw new Error('No projector loaded — call loadProjector() first');

    const release = await this.mutex.acquire();
    this.cachedTokenIds = [];
    const ctx = this.ensureHandle();
    const opts = mapGenerateOptions(this.native, options);

    try {
      yield* this.streamInternal(
        (cb) => this.native.inferStreamVision(this.model, ctx, this.mtmdCtx!, prompt, imageBuffers, opts, cb),
        options,
      );
    } finally {
      release();
    }
  }

  /**
   * Shared streaming generator — handles the queue, stop-sequence buffering,
   * and abort logic used by both stream() and streamVision().
   */
  private async *streamInternal(
    startInference: (cb: (error: Error | null, token: string | null) => void) => void,
    options?: GenerateOptions,
  ): AsyncGenerator<string> {
    const stop = options?.stop;
    const signal = options?.signal;
    const maxStopLen = stop?.length ? Math.max(...stop.map(s => s.length)) : 0;

    type QueueItem = { token: string } | { error: Error } | { done: true };
    const queue: QueueItem[] = [];
    let resolve: (() => void) | null = null;

    const notify = () => {
      if (resolve) {
        const r = resolve;
        resolve = null;
        r();
      }
    };

    const waitForItem = (): Promise<void> =>
      new Promise<void>((r) => {
        resolve = r;
      });

    const onAbort = () => { queue.push({ done: true }); notify(); };
    signal?.addEventListener('abort', onAbort, { once: true });

    startInference((error, token) => {
      if (signal?.aborted) return;
      if (error) {
        queue.push({ error });
      } else if (token === null) {
        queue.push({ done: true });
      } else {
        queue.push({ token });
      }
      notify();
    });

    let buffer = '';

    try {
      while (true) {
        if (queue.length === 0) {
          await waitForItem();
        }

        while (queue.length > 0) {
          const item = queue.shift()!;
          if ('error' in item) throw item.error;
          if ('done' in item) {
            if (stop?.length && buffer) {
              buffer = truncateAtStopSequence(buffer, stop);
              if (buffer) yield buffer;
            }
            return;
          }

          if (stop?.length) {
            buffer += item.token;

            const stopIdx = findStopSequence(buffer, stop);
            if (stopIdx !== -1) {
              const tail = buffer.slice(0, stopIdx);
              if (tail) yield tail;
              return;
            }

            if (buffer.length > maxStopLen) {
              const safe = buffer.length - maxStopLen;
              yield buffer.slice(0, safe);
              buffer = buffer.slice(safe);
            }
          } else {
            yield item.token;
          }
        }
      }
    } finally {
      signal?.removeEventListener('abort', onAbort);
    }
  }

  /**
   * Runs a streaming inference call and collects all tokens into a single string.
   * This avoids blocking the event loop (unlike inferSync which is synchronous C++).
   */
  private collectStream(
    start: (cb: (error: Error | null, token: string | null) => void) => void,
    signal?: AbortSignal,
  ): Promise<string> {
    return new Promise<string>((resolve, reject) => {
      const parts: string[] = [];

      const onAbort = () => resolve(parts.join(''));
      signal?.addEventListener('abort', onAbort, { once: true });

      start((error, token) => {
        if (signal?.aborted) return;
        if (error) {
          signal?.removeEventListener('abort', onAbort);
          reject(error);
        } else if (token === null) {
          signal?.removeEventListener('abort', onAbort);
          resolve(parts.join(''));
        } else {
          parts.push(token);
        }
      });
    });
  }

  async generateBatch(
    prompts: string[],
    options?: GenerateOptions | GenerateOptions[],
  ): Promise<BatchResult[]> {
    const release = await this.mutex.acquire();
    try {
      const ctx = this.ensureHandle();
      // Batch uses seq_ids, invalidate single-sequence cache
      this.cachedTokenIds = [];

      // Normalize options: single applies to all, array = per-prompt
      const optsList: GenerateOptions[] = Array.isArray(options)
        ? options
        : prompts.map(() => options ?? {});

      const nativeOpts = optsList.map((o) => mapGenerateOptions(this.native, o));

      const results: BatchResult[] = prompts.map((_, i) => ({
        index: i,
        text: '',
        finishReason: 'stop' as const,
      }));

      const stopLists = optsList.map((o) => o.stop ?? []);

      return new Promise<BatchResult[]>((resolve, reject) => {
        this.native.inferBatch(this.model, ctx, prompts, nativeOpts, (error, token, seqIndex, finishReason) => {
          if (error) {
            release();
            reject(error);
            return;
          }

          // seqIndex === -1 signals all sequences complete
          if (seqIndex === -1) {
            // Apply stop-sequence truncation per sequence
            for (let i = 0; i < results.length; i++) {
              if (stopLists[i].length > 0) {
                results[i].text = truncateAtStopSequence(results[i].text, stopLists[i]);
              }
            }
            release();
            resolve(results);
            return;
          }

          if (finishReason) {
            // Sequence completed
            results[seqIndex].finishReason = finishReason as 'stop' | 'length';
          } else if (token !== null) {
            // Token for this sequence
            results[seqIndex].text += token;
          }
        });
      });
    } catch (err) {
      release();
      throw wrapNativeError(err, 'inference');
    }
  }

  /** Run a single-token warmup pass to prime GPU shaders and CPU caches. */
  warmup(): void {
    this.native.warmup(this.model, this.ensureHandle());
  }

  /** Read performance metrics from the most recent inference call. */
  getPerf(): InferenceMetrics | null {
    return this.native.getPerf(this.ensureHandle());
  }

  /**
   * Run a benchmark entirely in the C engine: evaluate a synthetic prompt
   * then generate tokens, repeating for the requested number of iterations.
   * No FFI round-trips per iteration — timing is measured natively.
   */
  benchmark(options?: BenchmarkOptions): BenchmarkResult {
    this.cachedTokenIds = [];
    const result = this.native.benchmark(this.model, this.ensureHandle(), {
      promptTokens: options?.promptTokens,
      generateTokens: options?.generateTokens,
      iterations: options?.iterations,
    });
    return {
      promptTokensPerSec: result.promptTokensPerSec,
      generatedTokensPerSec: result.generatedTokensPerSec,
      ttftMs: result.ttftMs,
      totalMs: result.totalMs,
      iterations: result.iterations,
      individual: [],
    };
  }

  dispose(): void {
    this.cachedTokenIds = [];
    if (this.handle) {
      this.native.freeContext(this.handle);
      this.handle = null;
    }
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

/** Find the length of the common prefix between two arrays. */
function findCommonPrefixLength(a: number[], b: number[]): number {
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    if (a[i] !== b[i]) return i;
  }
  return len;
}

/** Well-known GBNF grammar for arbitrary JSON output. */
const JSON_GBNF = `root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws

object ::=
  "{" ws (
    string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws

array  ::=
  "[" ws (
    value
    ("," ws value)*
  )? "]" ws

string ::=
  "\\"" (
    [^\\\\\\x7F\\x00-\\x1F] |
    "\\\\" (["\\\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F])
  )* "\\"" ws

number ::= ("-"? ([0-9] | [1-9] [0-9]*)) ("." [0-9]+)? (("e" | "E") ("+" | "-")? [0-9]+)? ws

ws ::= ([ \\t\\n] ws)?
`;

function resolveGrammar(native: NativeAddon, options?: GenerateOptions): { grammar?: string; grammar_root?: string } {
  if (options?.grammar) {
    return { grammar: options.grammar, grammar_root: options.grammarRoot };
  }

  const fmt = options?.responseFormat;
  if (!fmt || fmt.type === 'text') return {};

  if (fmt.type === 'json_object') {
    return { grammar: JSON_GBNF };
  }

  if (fmt.type === 'json_schema' && fmt.json_schema?.schema) {
    const gbnf = native.jsonSchemaToGrammar(JSON.stringify(fmt.json_schema.schema));
    return { grammar: gbnf };
  }

  return {};
}

function mapGenerateOptions(native: NativeAddon, options?: GenerateOptions) {
  const grammarOpts = resolveGrammar(native, options);
  return defined({
    max_tokens: options?.maxTokens,
    temperature: options?.temperature,
    top_p: options?.topP,
    top_k: options?.topK,
    repeat_penalty: options?.repeatPenalty,
    frequency_penalty: options?.frequencyPenalty,
    presence_penalty: options?.presencePenalty,
    seed: options?.seed,
    grammar: grammarOpts.grammar,
    grammar_root: grammarOpts.grammar_root,
    onPromptProgress: options?.onPromptProgress,
  });
}

/**
 * Map compute mode to GPU layer count.
 * llama.cpp clamps n_gpu_layers to the model's actual layer count,
 * so 999 safely means "offload everything".
 */
function resolveCompute(mode: ComputeMode | undefined, native: NativeAddon): number {
  const resolved = (mode ?? 'auto') === 'auto' ? detectGpu(native) : mode!;
  switch (resolved) {
    case 'cpu': return 0;
    case 'hybrid': return 16;
    case 'gpu': return 999;
    default: return 999;
  }
}

/** Detect GPU availability from the native backend info. */
function detectGpu(native: NativeAddon): 'gpu' | 'cpu' {
  try {
    const info = native.backendInfo().toLowerCase();
    if (info.includes('metal') || info.includes('cuda') || info.includes('vulkan')) {
      return 'gpu';
    }
  } catch {
    // backendInfo unavailable — fall back to CPU
  }
  return 'cpu';
}

// ── Stop sequence helpers ───────────────────────────────────────────────────

function findStopSequence(text: string, stop: string[]): number {
  let earliest = -1;
  for (const seq of stop) {
    const idx = text.indexOf(seq);
    if (idx !== -1 && (earliest === -1 || idx < earliest)) {
      earliest = idx;
    }
  }
  return earliest;
}

function truncateAtStopSequence(text: string, stop: string[]): string {
  const idx = findStopSequence(text, stop);
  return idx === -1 ? text : text.slice(0, idx);
}

// ── Native error wrapping ───────────────────────────────────────────────────

type NativeErrorPhase = 'model-load' | 'context-create' | 'inference' | 'projector' | 'quantize';

function wrapNativeError(err: unknown, phase: NativeErrorPhase, path?: string): Error {
  const msg = err instanceof Error ? err.message : String(err);
  const lower = msg.toLowerCase();

  if (phase === 'model-load') {
    if (lower.includes('no such file') || lower.includes('not found') || lower.includes('enoent')) {
      return new Error(
        `Model file not found: ${path}\n` +
        `Ensure the path exists or use a valid HuggingFace shorthand (user/repo/file.gguf).`,
      );
    }
    if (lower.includes('alloc') || lower.includes('memory') || lower.includes('mmap failed')) {
      return new Error(
        `Failed to load model — insufficient memory.\n` +
        `Model: ${path}\n` +
        `Try a smaller quantization (e.g. Q4_K_M instead of Q8_0) or use compute: 'cpu' to avoid GPU memory limits.`,
      );
    }
    return new Error(`Failed to load model: ${msg}`);
  }

  if (phase === 'context-create') {
    if (lower.includes('alloc') || lower.includes('memory')) {
      return new Error(
        `Failed to create context — insufficient memory.\n` +
        `Try reducing contextSize or using a smaller model.`,
      );
    }
    return new Error(`Failed to create context: ${msg}`);
  }

  if (phase === 'projector') {
    if (lower.includes('no such file') || lower.includes('not found') || lower.includes('enoent')) {
      return new Error(
        `Projector file not found: ${path}\n` +
        `Ensure the mmproj GGUF file path is correct.`,
      );
    }
    return new Error(`Failed to load projector: ${msg}`);
  }

  if (phase === 'quantize') {
    if (lower.includes('no such file') || lower.includes('not found') || lower.includes('enoent')) {
      return new Error(
        `Model file not found: ${path}\n` +
        `Ensure the input GGUF file path is correct.`,
      );
    }
    return new Error(`Quantization failed: ${msg}`);
  }

  return new Error(`Inference failed: ${msg}`);
}

// ── Quantization ────────────────────────────────────────────────────────────

const QUANTIZE_FTYPE_MAP: Record<QuantizationType, number> = {
  F32: 0, F16: 1, Q4_0: 2, Q4_1: 3, Q8_0: 7, Q5_0: 8, Q5_1: 9,
  Q2_K: 10, Q3_K_S: 11, Q3_K_M: 12, Q3_K_L: 13,
  Q4_K_S: 14, Q4_K_M: 15, Q5_K_S: 16, Q5_K_M: 17, Q6_K: 18,
  IQ2_XXS: 19, IQ2_XS: 20, IQ3_XS: 22, IQ3_XXS: 23,
  IQ1_S: 24, IQ4_NL: 25, IQ3_S: 26, IQ3_M: 27, IQ2_S: 28, IQ2_M: 29,
  IQ4_XS: 30, IQ1_M: 31, BF16: 32, TQ1_0: 36, TQ2_0: 37,
};

/**
 * Quantize a GGUF model file to a lower-precision format.
 * This is a standalone function — it doesn't require a loaded Model.
 */
export async function quantize(input: string, output: string, options: QuantizeOptions): Promise<void> {
  const native = loadNativeAddon();
  const ftype = QUANTIZE_FTYPE_MAP[options.type];

  return new Promise<void>((resolve, reject) => {
    try {
      native.quantize(
        input,
        output,
        defined({
          ftype,
          nthread: options.threads,
          allow_requantize: options.allowRequantize,
          quantize_output_tensor: options.quantizeOutputTensor,
          pure: options.pure,
        }) as { ftype: number },
        (error) => {
          if (error) reject(wrapNativeError(error, 'quantize', input));
          else resolve();
        },
      );
    } catch (err) {
      reject(wrapNativeError(err, 'quantize', input));
    }
  });
}

// ── Flash attention / KV cache type mapping ─────────────────────────────────

function flashAttnToNative(mode: FlashAttentionMode): number {
  switch (mode) {
    case 'auto':     return -1; // LLAMA_FLASH_ATTN_TYPE_AUTO
    case 'disabled': return 0;  // LLAMA_FLASH_ATTN_TYPE_DISABLED
    case 'enabled':  return 1;  // LLAMA_FLASH_ATTN_TYPE_ENABLED
    default:         return -1;
  }
}

function kvCacheTypeToNative(type: KvCacheType): number {
  switch (type) {
    case 'f16':  return 1; // GGML_TYPE_F16
    case 'q4_0': return 2; // GGML_TYPE_Q4_0
    case 'q8_0': return 8; // GGML_TYPE_Q8_0
    default:     return 1;
  }
}

// ── Embedding context ──────────────────────────────────────────────────────

function poolingTypeToNative(pooling: EmbeddingPoolingType): number {
  switch (pooling) {
    case 'mean': return 1;
    case 'cls':  return 2;
    case 'last': return 3;
    default:     return 1;
  }
}

export class EmbeddingContext {
  private native: NativeAddon;
  private modelHandle: NativeModel;
  private handle: NativeContext | null;
  private _dimension: number;

  constructor(native: NativeAddon, modelHandle: NativeModel, handle: NativeContext, dimension: number) {
    this.native = native;
    this.modelHandle = modelHandle;
    this.handle = handle;
    this._dimension = dimension;
  }

  /** The dimensionality of embedding vectors produced by this model. */
  get dimension(): number {
    return this._dimension;
  }

  private ensureHandle(): NativeContext {
    if (!this.handle) throw new Error('EmbeddingContext has been disposed');
    return this.handle;
  }

  /** Embed a single text string. Returns a normalized Float32Array. */
  embed(text: string): Float32Array {
    const tokens = this.native.tokenize(this.modelHandle, text);
    return this.native.embed(this.ensureHandle(), this.modelHandle, tokens);
  }

  /**
   * Embed multiple texts in a single batch. More efficient than calling
   * embed() in a loop because all texts are encoded in one forward pass.
   */
  embedBatch(texts: string[]): Float32Array[] {
    const tokenArrays = texts.map((t) => this.native.tokenize(this.modelHandle, t));
    return this.native.embedBatch(this.ensureHandle(), this.modelHandle, tokenArrays);
  }

  dispose(): void {
    if (this.handle) {
      this.native.freeContext(this.handle);
      this.handle = null;
    }
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}

// ── Vision helpers ──────────────────────────────────────────────────────────

const MEDIA_MARKER = '<__media__>';

/** Resolve an image URL (data URI, file path, or HTTP URL) to a Buffer. */
export async function resolveImageToBuffer(url: string): Promise<Buffer> {
  // data: URI — base64 decode
  if (url.startsWith('data:')) {
    const commaIdx = url.indexOf(',');
    if (commaIdx === -1) throw new Error('Invalid data URI');
    const base64 = url.slice(commaIdx + 1);
    return Buffer.from(base64, 'base64');
  }

  // HTTP(S) URL
  if (url.startsWith('http://') || url.startsWith('https://')) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch image: ${res.status} ${res.statusText}`);
    return Buffer.from(await res.arrayBuffer());
  }

  // Local file path
  return readFile(url);
}

/**
 * Build a vision prompt from ChatMessages.
 * Extracts text with `<__media__>` markers for each image and returns
 * the ordered image buffers.
 */
export async function buildVisionPrompt(
  messages: ChatMessage[],
): Promise<{ text: string; imageBuffers: Buffer[] }> {
  const imageBuffers: Buffer[] = [];
  const textParts: string[] = [];

  for (const msg of messages) {
    if (typeof msg.content === 'string') {
      textParts.push(msg.content);
      continue;
    }

    // Array content — process parts
    const msgParts: string[] = [];
    for (const part of msg.content) {
      if (part.type === 'text') {
        msgParts.push(part.text);
      } else if (part.type === 'image_url') {
        const buf = await resolveImageToBuffer(part.image_url.url);
        imageBuffers.push(buf);
        msgParts.push(MEDIA_MARKER);
      }
    }
    textParts.push(msgParts.join(''));
  }

  return { text: textParts.join('\n'), imageBuffers };
}

/** Check if a list of messages contains any image content. */
export function hasImageContent(messages: Array<{ content: string | ContentPart[] | unknown }>): boolean {
  return messages.some((m) => {
    if (!Array.isArray(m.content)) return false;
    return (m.content as ContentPart[]).some((p) => p.type === 'image_url');
  });
}
