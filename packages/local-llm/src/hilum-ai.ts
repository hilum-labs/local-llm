import { BaseLocalLLM, type LocalLLMRuntimeAdapter, type LocalLLMOptions } from 'local-llm-js-core';
import { Model } from './engine.js';
import { ChatCompletions } from './openai-compat.js';
import { LocalLLMProvider } from './ai-sdk-provider.js';
import { ModelManager } from './model-manager.js';
import { loadNativeAddon } from './native.js';
import type { LogCallback, LogLevel } from './types.js';

const runtimeAdapter: LocalLLMRuntimeAdapter = {
  async loadModel(path, options) {
    return Model.load(path, options);
  },
  async resolveFilePath(path: string, options: LocalLLMOptions): Promise<string> {
    if (path.startsWith('/') || path.startsWith('./') || path.startsWith('../')) {
      return path;
    }

    const manager = new ModelManager(options.cacheDir);
    return manager.downloadModel(path, {
      onProgress: options.onProgress
        ? (_dl, _total, pct) => options.onProgress!(pct)
        : undefined,
    });
  },
  setLogCallback(callback: LogCallback | null, minLevel?: LogLevel): void {
    LocalLLM.setLogCallback(callback, minLevel);
  },
  createChatCompletions(model, context, modelName, config) {
    return new ChatCompletions(model, context, modelName, config);
  },
};

export type { LocalLLMOptions } from 'local-llm-js-core';

export class LocalLLM extends BaseLocalLLM {
  constructor(options: LocalLLMOptions) {
    super(options, runtimeAdapter);
  }

  /** Create and initialize a LocalLLM instance in one step. */
  static async create(options: LocalLLMOptions): Promise<LocalLLM> {
    const instance = new LocalLLM(options);
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
      addon.setLogLevel(LocalLLM.LOG_LEVEL_MAP[minLevel]);
      addon.setLogCallback((level: number, text: string) => {
        callback(LocalLLM.LOG_LEVEL_REVERSE[level] ?? 'info', text);
      });
    } else {
      addon.setLogCallback(null);
    }
  }

  /** Return a Vercel AI SDK LanguageModelV3 provider backed by this instance. */
  languageModel(id?: string): LocalLLMProvider {
    if (!this._model) {
      throw new Error('LocalLLM not initialized. Call await init() first, or use await LocalLLM.create().');
    }
    this.ensureContext();
    const modelId = id ?? this._modelName ?? 'local';
    return new LocalLLMProvider(this._model, this._context!, modelId);
  }
}
