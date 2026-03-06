import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { join } from 'node:path';
import { mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { ModelManager } from './model-manager.js';
import { ModelCache } from './cache.js';

// We test URL resolution and cache-hit logic without actually downloading.

describe('ModelManager', () => {
  let cacheDir: string;

  beforeEach(async () => {
    cacheDir = await mkdtemp(join(tmpdir(), 'local-llm-mm-test-'));
  });

  afterEach(async () => {
    await rm(cacheDir, { recursive: true, force: true });
  });

  describe('URL resolution (via resolveHuggingFaceUrl)', () => {
    // We test this indirectly through removeModel — it calls resolveHuggingFaceUrl internally

    it('full URLs pass through unchanged', async () => {
      const manager = new ModelManager(cacheDir);
      // removeModel calls resolveHuggingFaceUrl, so if it doesn't throw, the URL resolved
      const removed = await manager.removeModel('https://huggingface.co/user/repo/resolve/main/model.gguf');
      expect(removed).toBe(false); // nothing to remove, but URL was accepted
    });

    it('3-part shorthand resolves to HuggingFace URL', async () => {
      const manager = new ModelManager(cacheDir);
      // Pre-cache with the resolved URL so we can verify resolution
      const cache = new ModelCache(cacheDir);
      const resolvedUrl = 'https://huggingface.co/user/repo/resolve/main/model.gguf';
      const fakePath = join(cacheDir, 'model.gguf');
      await writeFile(fakePath, 'data');
      await cache.cacheModel(resolvedUrl, fakePath, 4);

      // removeModel with shorthand should resolve to the same URL and find the entry
      const removed = await manager.removeModel('user/repo/model.gguf');
      expect(removed).toBe(true);
    });

    it('4-part shorthand with branch resolves correctly', async () => {
      const manager = new ModelManager(cacheDir);
      const cache = new ModelCache(cacheDir);
      const resolvedUrl = 'https://huggingface.co/user/repo/resolve/dev/model.gguf';
      const fakePath = join(cacheDir, 'model.gguf');
      await writeFile(fakePath, 'data');
      await cache.cacheModel(resolvedUrl, fakePath, 4);

      const removed = await manager.removeModel('user/repo/dev/model.gguf');
      expect(removed).toBe(true);
    });

    it('invalid shorthand throws', async () => {
      const manager = new ModelManager(cacheDir);
      await expect(manager.removeModel('just-a-name')).rejects.toThrow('Invalid model specifier');
    });

    it('5-part shorthand throws', async () => {
      const manager = new ModelManager(cacheDir);
      await expect(manager.removeModel('a/b/c/d/e')).rejects.toThrow('Invalid model specifier');
    });
  });

  describe('cache hit', () => {
    it('returns cached path without downloading', async () => {
      const manager = new ModelManager(cacheDir);
      const cache = new ModelCache(cacheDir);
      const url = 'https://huggingface.co/user/repo/resolve/main/model.gguf';
      const fakePath = join(cacheDir, 'model.gguf');
      await writeFile(fakePath, 'cached model data');
      await cache.cacheModel(url, fakePath, 17);

      // downloadModel should return cached path without making a network request
      const result = await manager.downloadModel(url);
      expect(result).toBe(fakePath);
    });
  });

  describe('listModels', () => {
    it('returns empty list initially', async () => {
      const manager = new ModelManager(cacheDir);
      const list = await manager.listModels();
      expect(list).toEqual([]);
    });

    it('returns cached models', async () => {
      const manager = new ModelManager(cacheDir);
      const cache = new ModelCache(cacheDir);
      const url = 'https://example.com/model.gguf';
      const fakePath = join(cacheDir, 'model.gguf');
      await writeFile(fakePath, 'data');
      await cache.cacheModel(url, fakePath, 4);

      const list = await manager.listModels();
      expect(list).toHaveLength(1);
      expect(list[0].url).toBe(url);
    });
  });

  describe('retry behavior', () => {
    it('throws on 4xx errors without retry', async () => {
      const originalFetch = globalThis.fetch;
      globalThis.fetch = vi.fn(async () => new Response(null, { status: 404, statusText: 'Not Found' }));

      try {
        const manager = new ModelManager(cacheDir);
        await expect(
          manager.downloadModel('https://huggingface.co/user/repo/resolve/main/model.gguf'),
        ).rejects.toThrow('Download failed: 404');
        expect(globalThis.fetch).toHaveBeenCalledTimes(1);
      } finally {
        globalThis.fetch = originalFetch;
      }
    });

    it('retries on 5xx errors', async () => {
      const originalFetch = globalThis.fetch;
      let callCount = 0;
      globalThis.fetch = vi.fn(async () => {
        callCount++;
        return new Response(null, { status: 503, statusText: 'Service Unavailable' });
      });

      try {
        const manager = new ModelManager(cacheDir);
        await expect(
          manager.downloadModel('https://huggingface.co/user/repo/resolve/main/model.gguf'),
        ).rejects.toThrow('Download failed: 503');
        expect(callCount).toBeGreaterThan(1);
      } finally {
        globalThis.fetch = originalFetch;
      }
    }, 30000);
  });
});
