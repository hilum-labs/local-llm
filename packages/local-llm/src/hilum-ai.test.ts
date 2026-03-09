import { describe, it, expect, vi, beforeEach } from 'vitest';
import type { InferenceContext } from './engine.js';

const mockContext = {
  generate: vi.fn(async () => 'mock response'),
  stream: vi.fn(async function* () { yield 'hello'; }),
  generateVision: vi.fn(async () => 'mock vision'),
  streamVision: vi.fn(async function* () { yield 'img'; }),
  warmup: vi.fn(),
  dispose: vi.fn(),
};

const mockModel = {
  handle: {},
  sizeBytes: 1000,
  tokenize: vi.fn(() => new Int32Array(5)),
  detokenize: vi.fn(() => 'text'),
  applyChatTemplate: vi.fn(() => 'formatted-prompt'),
  createContext: vi.fn(() => mockContext),
  loadProjector: vi.fn(),
  isMultimodal: false,
  mtmdContext: null,
  dispose: vi.fn(),
};

vi.mock('./engine.js', () => {
  class Model {
    constructor(_path: string, _options?: any) {
      Object.assign(this, mockModel);
    }
    static async load(_path: string, _options?: any) {
      return Object.assign(Object.create(Model.prototype), mockModel);
    }
    static fromBuffer() {
      return Object.assign(Object.create(Model.prototype), mockModel);
    }
  }
  return { Model, InferenceContext: class {} };
});

vi.mock('./native.js', () => ({
  loadNativeAddon: vi.fn(() => ({
    setLogLevel: vi.fn(),
    setLogCallback: vi.fn(),
    backendInfo: vi.fn(() => 'Metal'),
  })),
}));

const mockDownloadModel = vi.fn(async () => '/cached/model.gguf');

vi.mock('./model-manager.js', () => {
  class ModelManager {
    downloadModel = mockDownloadModel;
  }
  return { ModelManager };
});

const { LocalLLM } = await import('./hilum-ai.js');

beforeEach(() => {
  vi.clearAllMocks();
  mockModel.createContext.mockReturnValue(mockContext);
});

describe('LocalLLM', () => {
  describe('constructor', () => {
    it('stores options without side effects', () => {
      const ai = new LocalLLM({ model: '/path/model.gguf' });
      expect(ai).toBeDefined();
    });
  });

  describe('chat getter', () => {
    it('throws before init', () => {
      const ai = new LocalLLM({ model: '/path/model.gguf' });
      expect(() => ai.chat).toThrow('not initialized');
    });

    it('returns completions after init', async () => {
      const ai = new LocalLLM({ model: '/path/model.gguf' });
      await ai.init();
      expect(ai.chat).toBeDefined();
      expect(ai.chat.completions).toBeDefined();
      ai.dispose();
    });
  });

  describe('init()', () => {
    it('loads a local model without downloading', async () => {
      const ai = new LocalLLM({ model: '/local/model.gguf' });
      await ai.init();

      expect(mockDownloadModel).not.toHaveBeenCalled();
      expect(ai.chat).toBeDefined();
      ai.dispose();
    });

    it('downloads a remote model via ModelManager', async () => {
      const ai = new LocalLLM({ model: 'user/repo/model.gguf' });
      await ai.init();

      expect(mockDownloadModel).toHaveBeenCalledWith(
        'user/repo/model.gguf',
        expect.any(Object),
      );
      ai.dispose();
    });

    it('is idempotent — second call is a no-op', async () => {
      const ai = new LocalLLM({ model: '/path/model.gguf' });
      await ai.init();
      await ai.init();

      // Context created once during init (warmup), not duplicated
      expect(mockModel.createContext).toHaveBeenCalledTimes(1);
      ai.dispose();
    });

    it('concurrent calls share the same init promise', async () => {
      const ai = new LocalLLM({ model: '/path/model.gguf' });
      const [a, b] = await Promise.all([ai.init(), ai.init()]);

      expect(a).toBeUndefined();
      expect(b).toBeUndefined();
      expect(mockModel.createContext).toHaveBeenCalledTimes(1);
      ai.dispose();
    });

    it('passes context options to createContext', async () => {
      const ai = new LocalLLM({
        model: '/path/model.gguf',
        contextSize: 4096,
        batchSize: 512,
        threads: 4,
      });
      await ai.init();

      expect(mockModel.createContext).toHaveBeenCalledWith(
        expect.objectContaining({
          contextSize: 4096,
          batchSize: 512,
          threads: 4,
        }),
      );
      ai.dispose();
    });
  });

  describe('create()', () => {
    it('returns an initialized instance', async () => {
      const ai = await LocalLLM.create({ model: '/path/model.gguf' });
      expect(ai.chat).toBeDefined();
      ai.dispose();
    });
  });

  describe('dispose()', () => {
    it('disposes context and model', async () => {
      const ai = await LocalLLM.create({ model: '/path/model.gguf' });
      // Trigger lazy context creation so there's something to dispose
      void ai.chat;
      ai.dispose();

      expect(mockContext.dispose).toHaveBeenCalled();
      expect(mockModel.dispose).toHaveBeenCalled();
    });

    it('nulls out chat so getter throws after dispose', async () => {
      const ai = await LocalLLM.create({ model: '/path/model.gguf' });
      ai.dispose();

      expect(() => ai.chat).toThrow('not initialized');
    });

    it('is safe to call multiple times', async () => {
      const ai = await LocalLLM.create({ model: '/path/model.gguf' });
      ai.dispose();
      ai.dispose();
    });
  });

  describe('languageModel()', () => {
    it('throws before init', () => {
      const ai = new LocalLLM({ model: '/path/model.gguf' });
      expect(() => ai.languageModel()).toThrow('not initialized');
    });

    it('returns a language model provider after init', async () => {
      const ai = await LocalLLM.create({ model: '/path/model.gguf' });
      const lm = ai.languageModel();
      expect(lm).toBeDefined();
      ai.dispose();
    });

    it('uses custom model id when provided', async () => {
      const ai = await LocalLLM.create({ model: '/path/model.gguf' });
      const lm = ai.languageModel('my-model');
      expect(lm).toBeDefined();
      ai.dispose();
    });
  });

  describe('preload()', () => {
    it('returns local path without downloading', async () => {
      const result = await LocalLLM.preload('/local/model.gguf');
      expect(result).toBe('/local/model.gguf');
    });

    it('returns local path for relative paths', async () => {
      const result = await LocalLLM.preload('./model.gguf');
      expect(result).toBe('./model.gguf');
    });

    it('downloads remote model and returns cached path', async () => {
      const result = await LocalLLM.preload('user/repo/model.gguf');
      expect(result).toBe('/cached/model.gguf');
    });
  });

  describe('speculative decoding', () => {
    it('loads draft model and passes to createContext', async () => {
      const ai = await LocalLLM.create({
        model: '/path/model.gguf',
        draftModel: '/path/draft.gguf',
        draftNMax: 8,
      });

      expect(mockModel.createContext).toHaveBeenCalledWith(
        expect.objectContaining({ draftNMax: 8 }),
      );
      ai.dispose();
    });

    it('works without draft model', async () => {
      const ai = await LocalLLM.create({ model: '/path/model.gguf' });
      expect(ai.chat).toBeDefined();
      ai.dispose();
    });
  });

  describe('pool', () => {
    it('returns a ModelPool singleton', () => {
      const pool1 = LocalLLM.pool;
      const pool2 = LocalLLM.pool;
      expect(pool1).toBe(pool2);
    });
  });
});
