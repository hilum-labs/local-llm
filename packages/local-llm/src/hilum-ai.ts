import { Model } from './engine.js';
import type { InferenceContext, EmbeddingContext } from './engine.js';
import { ChatCompletions } from './openai-compat.js';
import { Embeddings } from './embeddings.js';
import { LocalAILanguageModel } from './ai-sdk-provider.js';
import { ModelManager } from './model-manager.js';
import { ModelPool } from './model-pool.js';
import { loadNativeAddon } from './native.js';
import type { ComputeMode, LogLevel, LogCallback, ContextOverflowStrategy, ContextOverflowConfig, EmbeddingPoolingType, FlashAttentionMode, KvCacheType } from './types.js';

export interface LocalAIOptions {
  /** HuggingFace URL, shorthand ("user/repo/file.gguf"), or local file path. */
  model: string;
  /** Path to mmproj GGUF file (required for vision models). URL, shorthand, or local path. */
  projector?: string;
  /** Compute mode: 'auto' (default — detects GPU), 'cpu', 'hybrid', or 'gpu'. */
  compute?: ComputeMode;
  /** Override GPU layer count directly (advanced — prefer `compute`). */
  gpuLayers?: number;
  useMmap?: boolean;
  contextSize?: number;
  batchSize?: number;
  threads?: number;
  /** Flash attention mode: 'auto' (default), 'enabled', 'disabled'. */
  flashAttention?: FlashAttentionMode;
  /** KV cache key quantization type: 'f16' (default), 'q8_0', 'q4_0'. Requires flash attention. */
  kvCacheTypeK?: KvCacheType;
  /** KV cache value quantization type: 'f16' (default), 'q8_0', 'q4_0'. Requires flash attention. */
  kvCacheTypeV?: KvCacheType;
  /** Cache directory for downloaded models (default: ~/.local-llm/models). */
  cacheDir?: string;
  /** Progress callback during model download (0–100). */
  onProgress?: (percent: number) => void;
  /** Log callback — receives llama.cpp log messages with level. */
  onLog?: LogCallback;
  /** Minimum log level to forward (default: 'info'). */
  logLevel?: LogLevel;
  /**
   * Context overflow strategy or full config.
   * Pass a string ('error' | 'truncate_oldest' | 'sliding_window') for simple usage,
   * or a ContextOverflowConfig for reserve ratio, onOverflow callback, etc.
   * Default: 'sliding_window' with 25% reserve ratio.
   */
  contextOverflow?: ContextOverflowStrategy | ContextOverflowConfig;
  /**
   * Enable the embeddings API. When true, creates an embedding context alongside
   * the inference context. The pooling type controls how token vectors are combined.
   * Default pooling: 'mean'.
   */
  embeddings?: boolean | { poolingType?: EmbeddingPoolingType };
}

export class LocalAI {
  private options: LocalAIOptions;
  private _model: Model | null = null;
  private _context: InferenceContext | null = null;
  private _embeddingContext: EmbeddingContext | null = null;
  private _chat: { completions: ChatCompletions } | null = null;
  private _embeddings: Embeddings | null = null;
  private _modelName: string | null = null;
  private _initPromise: Promise<void> | null = null;

  private static _pool: ModelPool | null = null;

  /** Shared model pool for in-process multimodel support. */
  static get pool(): ModelPool {
    if (!LocalAI._pool) {
      LocalAI._pool = new ModelPool();
    }
    return LocalAI._pool;
  }

  /** OpenAI-compatible chat completions interface. Lazily creates context on first use. */
  get chat(): { completions: ChatCompletions } {
    if (this._chat) return this._chat;
    if (this._model) {
      this.ensureContext();
      return this._chat!;
    }
    throw new Error('LocalAI not initialized. Call await init() first, or use await LocalAI.create().');
  }

  /** OpenAI-compatible embeddings interface. Requires `embeddings: true` in options. */
  get embeddings(): Embeddings {
    if (this._embeddings) return this._embeddings;
    if (this._model && !this._embeddingContext) {
      throw new Error('Embeddings not enabled. Pass `embeddings: true` to LocalAI.create().');
    }
    throw new Error('LocalAI not initialized. Call await init() first, or use await LocalAI.create().');
  }

  constructor(options: LocalAIOptions) {
    this.options = options;
  }

  /** Create and initialize a LocalAI instance in one step. */
  static async create(options: LocalAIOptions): Promise<LocalAI> {
    const instance = new LocalAI(options);
    await instance.init();
    return instance;
  }

  /**
   * Pre-download a model to the local cache without loading it into memory.
   * Call this early (e.g. at app startup) so that `create()` later is instant.
   * Returns the cached file path. Safe to call multiple times — cached models resolve immediately.
   */
  static async preload(
    model: string,
    options?: { cacheDir?: string; projector?: string; onProgress?: (percent: number) => void },
  ): Promise<string> {
    const manager = new ModelManager(options?.cacheDir);
    const progressCb = options?.onProgress
      ? (_dl: number, _total: number, pct: number) => options.onProgress!(pct)
      : undefined;

    // Start projector download in parallel if it's a remote path
    const isRemoteProjector = options?.projector &&
      !options.projector.startsWith('/') &&
      !options.projector.startsWith('./') &&
      !options.projector.startsWith('../');

    const projectorPromise = isRemoteProjector
      ? manager.downloadModel(options!.projector!, { onProgress: progressCb })
      : undefined;

    // Local file path — wait for projector (if any) and return
    if (model.startsWith('/') || model.startsWith('./') || model.startsWith('../')) {
      await projectorPromise;
      return model;
    }

    const [modelPath] = await Promise.all(
      projectorPromise
        ? [manager.downloadModel(model, { onProgress: progressCb }), projectorPromise]
        : [manager.downloadModel(model, { onProgress: progressCb })],
    );

    return modelPath;
  }

  /** Initialize: download model (if URL) and load into memory. */
  async init(): Promise<void> {
    if (this._model) return; // already initialized
    if (this._initPromise) return this._initPromise;

    this._initPromise = this._doInit();
    return this._initPromise;
  }

  private static readonly LOG_LEVEL_MAP: Record<LogLevel, number> = {
    debug: 1,
    info: 2,
    warn: 3,
    error: 4,
  };

  private static readonly LOG_LEVEL_REVERSE: Record<number, LogLevel> = {
    1: 'debug',
    2: 'info',
    3: 'warn',
    4: 'error',
  };

  /** Set a log callback that receives llama.cpp log messages. Pass null to clear. */
  static setLogCallback(callback: LogCallback | null, minLevel: LogLevel = 'info'): void {
    const addon = loadNativeAddon();
    if (callback) {
      addon.setLogLevel(LocalAI.LOG_LEVEL_MAP[minLevel]);
      addon.setLogCallback((level: number, text: string) => {
        callback(LocalAI.LOG_LEVEL_REVERSE[level] ?? 'info', text);
      });
    } else {
      addon.setLogCallback(null);
    }
  }

  private async _doInit(): Promise<void> {
    // Wire up log callback if provided
    if (this.options.onLog) {
      LocalAI.setLogCallback(this.options.onLog, this.options.logLevel ?? 'info');
    }

    const modelPath = await this.resolveModelPath();

    this._model = await Model.load(modelPath, {
      compute: this.options.compute,
      gpuLayers: this.options.gpuLayers,
      useMmap: this.options.useMmap,
    });

    // Load projector for vision models
    if (this.options.projector) {
      const projectorPath = await this.resolveFilePath(this.options.projector);
      this._model.loadProjector(projectorPath, {
        threads: this.options.threads,
      });
    }

    this._modelName = modelPath.split('/').pop()?.replace('.gguf', '') ?? 'local';

    // Context and ChatCompletions are created lazily on first chat access
    // to save ~50-200MB when the model is loaded but not immediately used.

    if (this.options.embeddings) {
      const embOpts = typeof this.options.embeddings === 'object' ? this.options.embeddings : {};
      this._embeddingContext = this._model.createEmbeddingContext({
        contextSize: this.options.contextSize,
        batchSize: this.options.batchSize,
        threads: this.options.threads,
        poolingType: embOpts.poolingType,
      });
      this._embeddings = new Embeddings(this._model, this._embeddingContext, this._modelName);
    }
  }

  /** Lazily create inference context and ChatCompletions on first use. */
  private ensureContext(): void {
    if (this._context) return;
    if (!this._model) throw new Error('LocalAI not initialized.');

    this._context = this._model.createContext({
      contextSize: this.options.contextSize,
      batchSize: this.options.batchSize,
      threads: this.options.threads,
      flashAttention: this.options.flashAttention,
      kvCacheTypeK: this.options.kvCacheTypeK,
      kvCacheTypeV: this.options.kvCacheTypeV,
    });

    this._chat = {
      completions: new ChatCompletions(this._model, this._context, this._modelName ?? 'local', {
        contextOverflow: this.options.contextOverflow,
      }),
    };
  }

  private async resolveModelPath(): Promise<string> {
    return this.resolveFilePath(this.options.model);
  }

  private async resolveFilePath(path: string): Promise<string> {
    // Local file path — use directly
    if (path.startsWith('/') || path.startsWith('./') || path.startsWith('../')) {
      return path;
    }

    // URL or shorthand (e.g. "user/repo/file.gguf") — download via ModelManager
    const manager = new ModelManager(this.options.cacheDir);
    return manager.downloadModel(path, {
      onProgress: this.options.onProgress
        ? (_dl, _total, pct) => this.options.onProgress!(pct)
        : undefined,
    });
  }

  /** Switch to a model already loaded in the shared pool. */
  async switchModel(alias: string): Promise<void> {
    const model = LocalAI.pool.get(alias);
    if (!model) {
      throw new Error(`Model "${alias}" not found in pool. Load it first with LocalAI.pool.load().`);
    }

    // Dispose old context but not the old model (pool manages model lifecycle)
    this._context?.dispose();

    this._model = model;
    this._context = model.createContext({
      contextSize: this.options.contextSize,
      batchSize: this.options.batchSize,
      threads: this.options.threads,
      flashAttention: this.options.flashAttention,
      kvCacheTypeK: this.options.kvCacheTypeK,
      kvCacheTypeV: this.options.kvCacheTypeV,
    });

    this._modelName = alias;
    this._chat = {
      completions: new ChatCompletions(this._model, this._context, this._modelName, {
        contextOverflow: this.options.contextOverflow,
      }),
    };
  }

  /** Return a Vercel AI SDK LanguageModelV3 provider backed by this instance. */
  languageModel(id?: string): LocalAILanguageModel {
    if (!this._model) {
      throw new Error('LocalAI not initialized. Call await init() first, or use await LocalAI.create().');
    }
    this.ensureContext();
    const modelId = id ?? this._modelName ?? 'local';
    return new LocalAILanguageModel(this._model, this._context!, modelId);
  }

  dispose(): void {
    this._embeddingContext?.dispose();
    this._context?.dispose();
    this._model?.dispose();
    this._embeddingContext = null;
    this._embeddings = null;
    this._context = null;
    this._model = null;
    this._chat = null;
  }

  [Symbol.dispose](): void {
    this.dispose();
  }
}
