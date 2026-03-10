import { describe, it, expect, vi, beforeEach } from 'vitest';
import { setNativeAddon, type NativeAddon, type NativeModel } from 'local-llm-js-core';
import { ModelPool } from './model-pool.js';

function createHandle(sizeBytes: number): NativeModel & { sizeBytes: number } {
  return { __brand: 'NativeModel', sizeBytes } as NativeModel & { sizeBytes: number };
}

const mockNative: NativeAddon = {
  backendInfo: vi.fn(() => 'Test'),
  backendVersion: vi.fn(() => '1.0.0'),
  apiVersion: vi.fn(() => 1 << 16),
  loadModel: vi.fn((path: string) => {
    const match = path.match(/model-(\d+)/);
    return createHandle(match ? Number(match[1]) : 1000);
  }),
  loadModelFromBuffer: vi.fn((buffer: Uint8Array) => createHandle(buffer.byteLength || 1000)),
  createContext: vi.fn(),
  tokenize: vi.fn(() => new Int32Array()),
  detokenize: vi.fn(() => ''),
  applyChatTemplate: vi.fn(() => ''),
  inferSync: vi.fn(() => ''),
  inferStream: vi.fn(),
  getModelSize: vi.fn((model: NativeModel) => (model as NativeModel & { sizeBytes: number }).sizeBytes),
  freeModel: vi.fn(),
  freeContext: vi.fn(),
  getContextSize: vi.fn(() => 0),
  warmup: vi.fn(),
  kvCacheClear: vi.fn(),
  stopStream: vi.fn(),
  getPerf: vi.fn(() => null),
  createMtmdContext: vi.fn(),
  freeMtmdContext: vi.fn(),
  supportVision: vi.fn(() => false),
  inferSyncVision: vi.fn(() => ''),
  inferStreamVision: vi.fn(),
  inferBatch: vi.fn(),
  jsonSchemaToGrammar: vi.fn(() => ''),
  getEmbeddingDimension: vi.fn(() => 0),
  createEmbeddingContext: vi.fn(),
  embed: vi.fn(),
  embedBatch: vi.fn(),
  quantize: vi.fn(),
  setLogCallback: vi.fn(),
  setLogLevel: vi.fn(),
  optimalThreadCount: vi.fn(() => 4),
  benchmark: vi.fn(),
};

beforeEach(() => {
  vi.clearAllMocks();
  setNativeAddon(mockNative);
});

describe('ModelPool', () => {
  describe('load / get / release lifecycle', () => {
    it('loads a model and retrieves it by alias', async () => {
      const pool = new ModelPool();
      const model = await pool.load('chat', 'model-1000.gguf');

      expect(model).toBeDefined();
      expect(pool.get('chat')).toBe(model);

      pool.dispose();
    });

    it('returns the same model on duplicate load (increments refCount)', async () => {
      const pool = new ModelPool();
      const model1 = await pool.load('chat', 'model-1000.gguf');
      const model2 = await pool.load('chat', 'model-1000.gguf');

      expect(model1).toBe(model2);

      const info = pool.list();
      expect(info[0].refCount).toBe(2);

      pool.dispose();
    });

    it('release decrements refCount; disposes at 0', async () => {
      const pool = new ModelPool();
      const model = await pool.load('chat', 'model-1000.gguf');

      // Load a second time to bump refCount to 2
      await pool.load('chat', 'model-1000.gguf');
      expect(pool.list()[0].refCount).toBe(2);

      pool.release('chat');
      expect(pool.list()[0].refCount).toBe(1);

      pool.release('chat');
      // refCount reached 0 — model removed from pool
      expect(pool.list()).toHaveLength(0);
      expect(pool.get('chat')).toBeUndefined();
    });

    it('release returns false for unknown alias', () => {
      const pool = new ModelPool();
      expect(pool.release('nonexistent')).toBe(false);
    });
  });

  describe('unload', () => {
    it('force unloads regardless of refCount', async () => {
      const pool = new ModelPool();
      await pool.load('chat', 'model-1000.gguf');
      await pool.load('chat', 'model-1000.gguf'); // refCount = 2

      pool.unload('chat');
      expect(pool.list()).toHaveLength(0);
      expect(pool.get('chat')).toBeUndefined();
    });

    it('does nothing for unknown alias', () => {
      const pool = new ModelPool();
      pool.unload('nonexistent'); // should not throw
    });
  });

  describe('list', () => {
    it('returns pool info for all loaded models', async () => {
      const pool = new ModelPool();
      await pool.load('chat', 'model-1000.gguf');
      await pool.load('code', 'model-2000.gguf');

      const info = pool.list();
      expect(info).toHaveLength(2);

      const aliases = info.map((i) => i.alias).sort();
      expect(aliases).toEqual(['chat', 'code']);

      for (const entry of info) {
        expect(entry.sizeBytes).toBeTypeOf('number');
        expect(entry.refCount).toBe(1);
        expect(entry.lastAccess).toBeTypeOf('number');
      }

      pool.dispose();
    });
  });

  describe('LRU eviction by maxModels', () => {
    it('evicts oldest unreferenced model when maxModels exceeded', async () => {
      const pool = new ModelPool({ maxModels: 2 });

      const m1 = await pool.load('a', 'model-100.gguf');
      const m2 = await pool.load('b', 'model-100.gguf');

      // Release 'a' so it's eligible for eviction
      pool.release('a');

      // Loading a third model should evict 'a'
      const m3 = await pool.load('c', 'model-100.gguf');

      const aliases = pool.list().map((i) => i.alias).sort();
      expect(aliases).toEqual(['b', 'c']);

      pool.dispose();
    });

    it('does not evict models with refCount > 0', async () => {
      const pool = new ModelPool({ maxModels: 2 });

      await pool.load('a', 'model-100.gguf');
      await pool.load('b', 'model-100.gguf');

      // Both have refCount = 1, so eviction fails — pool may exceed maxModels
      const m3 = await pool.load('c', 'model-100.gguf');

      // All three should still be present since none could be evicted
      expect(pool.list()).toHaveLength(3);

      pool.dispose();
    });
  });

  describe('LRU eviction by maxMemoryBytes', () => {
    it('evicts models to stay within memory limit', async () => {
      const pool = new ModelPool({ maxMemoryBytes: 2500 });

      await pool.load('a', 'model-1000.gguf');
      await pool.load('b', 'model-1000.gguf');

      // Release 'a' to make it evictable
      pool.release('a');

      // Loading 'c' (1000 bytes) would push total to 3000 — should evict 'a'
      await pool.load('c', 'model-1000.gguf');

      const aliases = pool.list().map((i) => i.alias).sort();
      expect(aliases).toEqual(['b', 'c']);

      pool.dispose();
    });
  });

  describe('dispose', () => {
    it('disposes all models and clears the pool', async () => {
      const pool = new ModelPool();
      await pool.load('a', 'model-100.gguf');
      await pool.load('b', 'model-100.gguf');

      pool.dispose();

      expect(pool.list()).toHaveLength(0);
    });

    it('is safe to call multiple times', async () => {
      const pool = new ModelPool();
      await pool.load('a', 'model-100.gguf');

      pool.dispose();
      pool.dispose(); // should not throw
      expect(pool.list()).toHaveLength(0);
    });
  });

  describe('get updates lastAccess', () => {
    it('updates lastAccess timestamp on get', async () => {
      const pool = new ModelPool();
      await pool.load('chat', 'model-1000.gguf');
      const info1 = pool.list()[0];
      const t1 = info1.lastAccess;

      // Small delay to ensure different timestamp
      await new Promise((r) => setTimeout(r, 10));

      pool.get('chat');
      const info2 = pool.list()[0];
      expect(info2.lastAccess).toBeGreaterThanOrEqual(t1);

      pool.dispose();
    });
  });
});
