import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { join } from 'node:path';
import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { writeFile, mkdir } from 'node:fs/promises';
import { ModelCache } from './cache.js';

describe('ModelCache', () => {
  let cacheDir: string;
  let cache: ModelCache;

  beforeEach(async () => {
    cacheDir = await mkdtemp(join(tmpdir(), 'local-llm-cache-test-'));
    cache = new ModelCache(cacheDir);
  });

  afterEach(async () => {
    await rm(cacheDir, { recursive: true, force: true });
  });

  describe('hashUrl', () => {
    it('produces a stable 16-char hex hash', () => {
      const hash = ModelCache.hashUrl('https://example.com/model.gguf');
      expect(hash).toMatch(/^[a-f0-9]{16}$/);
    });

    it('returns the same hash for the same URL', () => {
      const a = ModelCache.hashUrl('https://example.com/model.gguf');
      const b = ModelCache.hashUrl('https://example.com/model.gguf');
      expect(a).toBe(b);
    });

    it('returns different hashes for different URLs', () => {
      const a = ModelCache.hashUrl('https://example.com/model-a.gguf');
      const b = ModelCache.hashUrl('https://example.com/model-b.gguf');
      expect(a).not.toBe(b);
    });
  });

  describe('cacheModel / getCachedModel', () => {
    it('writes and reads a cache entry', async () => {
      const url = 'https://example.com/model.gguf';
      const filePath = join(cacheDir, 'fake-model.gguf');
      await writeFile(filePath, 'fake model data');

      await cache.cacheModel(url, filePath, 15);

      const entry = await cache.getCachedModel(url);
      expect(entry).not.toBeNull();
      expect(entry!.url).toBe(url);
      expect(entry!.path).toBe(filePath);
      expect(entry!.size).toBe(15);
      expect(entry!.downloadedAt).toBeTruthy();
      expect(entry!.lastUsedAt).toBeTruthy();
    });

    it('returns null for uncached URL', async () => {
      const entry = await cache.getCachedModel('https://example.com/not-cached.gguf');
      expect(entry).toBeNull();
    });

    it('returns null and cleans index when file was deleted externally', async () => {
      const url = 'https://example.com/model.gguf';
      const filePath = join(cacheDir, 'will-be-deleted.gguf');
      await writeFile(filePath, 'data');
      await cache.cacheModel(url, filePath, 4);

      // Delete the file externally
      await rm(filePath);

      const entry = await cache.getCachedModel(url);
      expect(entry).toBeNull();
    });

    it('creates the cache directory if it does not exist', async () => {
      const nestedDir = join(cacheDir, 'nested', 'deep');
      const nestedCache = new ModelCache(nestedDir);
      const url = 'https://example.com/model.gguf';
      const filePath = join(cacheDir, 'model.gguf');
      await writeFile(filePath, 'data');

      await nestedCache.cacheModel(url, filePath, 4);

      // Index should be written
      const indexPath = join(nestedDir, 'index.json');
      const indexContent = await readFile(indexPath, 'utf-8');
      expect(JSON.parse(indexContent).models).toBeTruthy();
    });

    it('overwrites entry when same URL is cached again', async () => {
      const url = 'https://example.com/model.gguf';
      const path1 = join(cacheDir, 'v1.gguf');
      const path2 = join(cacheDir, 'v2.gguf');
      await writeFile(path1, 'v1');
      await writeFile(path2, 'v2');

      await cache.cacheModel(url, path1, 2);
      await cache.cacheModel(url, path2, 2);

      const entry = await cache.getCachedModel(url);
      expect(entry!.path).toBe(path2);
    });
  });

  describe('listModels', () => {
    it('returns empty array when no models cached', async () => {
      const list = await cache.listModels();
      expect(list).toEqual([]);
    });

    it('returns all cached models', async () => {
      const file1 = join(cacheDir, 'a.gguf');
      const file2 = join(cacheDir, 'b.gguf');
      await writeFile(file1, 'a');
      await writeFile(file2, 'b');

      await cache.cacheModel('https://example.com/a.gguf', file1, 1);
      await cache.cacheModel('https://example.com/b.gguf', file2, 1);

      const list = await cache.listModels();
      expect(list).toHaveLength(2);
      const urls = list.map((e) => e.url).sort();
      expect(urls).toEqual([
        'https://example.com/a.gguf',
        'https://example.com/b.gguf',
      ]);
    });
  });

  describe('removeModel', () => {
    it('removes a cached model and returns true', async () => {
      const url = 'https://example.com/model.gguf';
      const hash = ModelCache.hashUrl(url);
      const modelDir = join(cacheDir, hash);
      await mkdir(modelDir, { recursive: true });
      const filePath = join(modelDir, 'model.gguf');
      await writeFile(filePath, 'data');

      await cache.cacheModel(url, filePath, 4);

      const removed = await cache.removeModel(url);
      expect(removed).toBe(true);

      const entry = await cache.getCachedModel(url);
      expect(entry).toBeNull();
    });

    it('returns false for uncached URL', async () => {
      const removed = await cache.removeModel('https://example.com/nope.gguf');
      expect(removed).toBe(false);
    });
  });

  describe('modelDir', () => {
    it('returns path under cacheDir using hash', () => {
      const url = 'https://example.com/model.gguf';
      const dir = cache.modelDir(url);
      expect(dir).toBe(join(cacheDir, ModelCache.hashUrl(url)));
    });
  });

  describe('corrupt index handling', () => {
    it('recovers from corrupt JSON and creates backup', async () => {
      const indexPath = join(cacheDir, 'index.json');
      await mkdir(cacheDir, { recursive: true });
      await writeFile(indexPath, '{ invalid json !!!');

      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const list = await cache.listModels();
      expect(list).toEqual([]);
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('Cache index was corrupt'),
      );

      warnSpy.mockRestore();
    });

    it('still works normally after recovery from corruption', async () => {
      const indexPath = join(cacheDir, 'index.json');
      await mkdir(cacheDir, { recursive: true });
      await writeFile(indexPath, 'not valid json');

      const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});

      const url = 'https://example.com/model.gguf';
      const filePath = join(cacheDir, 'model.gguf');
      await writeFile(filePath, 'model data');

      await cache.cacheModel(url, filePath, 10);
      const entry = await cache.getCachedModel(url);
      expect(entry).not.toBeNull();
      expect(entry!.url).toBe(url);

      warnSpy.mockRestore();
    });

    it('returns empty for missing index file (not an error)', async () => {
      const freshCache = new ModelCache(join(cacheDir, 'nonexistent'));
      const list = await freshCache.listModels();
      expect(list).toEqual([]);
    });
  });
});
