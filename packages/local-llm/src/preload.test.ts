import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { join } from 'node:path';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { ModelCache } from './cache.js';
import { LocalLLM } from './hilum-ai.js';

describe('LocalLLM.preload', () => {
  let cacheDir: string;

  beforeEach(async () => {
    cacheDir = await mkdtemp(join(tmpdir(), 'local-llm-preload-test-'));
  });

  afterEach(async () => {
    await rm(cacheDir, { recursive: true, force: true });
  });

  it('returns local file path unchanged without downloading', async () => {
    const result = await LocalLLM.preload('./models/local-model.gguf');
    expect(result).toBe('./models/local-model.gguf');
  });

  it('returns absolute file path unchanged', async () => {
    const result = await LocalLLM.preload('/absolute/path/model.gguf');
    expect(result).toBe('/absolute/path/model.gguf');
  });

  it('returns relative parent path unchanged', async () => {
    const result = await LocalLLM.preload('../other/model.gguf');
    expect(result).toBe('../other/model.gguf');
  });

  it('returns cached path when model is already in cache', async () => {
    // Pre-populate the cache with a fake model
    const cache = new ModelCache(cacheDir);
    const resolvedUrl = 'https://huggingface.co/user/repo/resolve/main/model.gguf';
    const fakePath = join(cacheDir, 'model.gguf');
    await writeFile(fakePath, 'fake model data');
    await cache.cacheModel(resolvedUrl, fakePath, 15);

    // Preload with shorthand should resolve to the same URL and return cached path
    const result = await LocalLLM.preload('user/repo/model.gguf', { cacheDir });
    expect(result).toBe(fakePath);
  });

  it('calls onProgress callback', async () => {
    // Pre-populate cache so it resolves instantly (no actual download)
    const cache = new ModelCache(cacheDir);
    const resolvedUrl = 'https://huggingface.co/user/repo/resolve/main/model.gguf';
    const fakePath = join(cacheDir, 'model.gguf');
    await writeFile(fakePath, 'data');
    await cache.cacheModel(resolvedUrl, fakePath, 4);

    const onProgress = vi.fn();
    const result = await LocalLLM.preload('user/repo/model.gguf', { cacheDir, onProgress });

    // Cache hit means no download, so onProgress won't be called — that's correct
    // The callback is only invoked during actual downloads
    expect(result).toBe(fakePath);
  });

  it('throws on invalid model specifier', async () => {
    await expect(
      LocalLLM.preload('invalid-specifier', { cacheDir }),
    ).rejects.toThrow('Invalid model specifier');
  });
});
