import { createWriteStream } from 'node:fs';
import { mkdir, stat, rename, unlink } from 'node:fs/promises';
import { join } from 'node:path';
import { pipeline } from 'node:stream/promises';
import { Readable } from 'node:stream';
import { ModelCache, type CacheEntry } from './cache.js';
import type { DownloadAdapter } from './download-adapter.js';

const MAX_RETRIES = 3;
const BASE_DELAY_MS = 1000;

export interface DownloadOptions {
  onProgress?: (downloaded: number, total: number, percent: number) => void;
}

export class ModelManager {
  private cache: ModelCache;
  private downloader: DownloadAdapter | null;

  constructor(cacheDir?: string, downloader?: DownloadAdapter) {
    this.cache = new ModelCache(cacheDir);
    this.downloader = downloader ?? null;
  }

  /**
   * Download a GGUF model from a URL (typically HuggingFace).
   * Returns the local file path. Uses cache — second call is instant.
   * Retries up to 3 times on transient network errors or server failures.
   */
  async downloadModel(url: string, options?: DownloadOptions): Promise<string> {
    const resolvedUrl = resolveHuggingFaceUrl(url);

    // Check cache first
    const cached = await this.cache.getCachedModel(resolvedUrl);
    if (cached) return cached.path;

    // Prepare target directory
    const dir = this.cache.modelDir(resolvedUrl);
    await mkdir(dir, { recursive: true });

    const filename = extractFilename(resolvedUrl);
    const targetPath = join(dir, filename);
    const tmpPath = targetPath + '.download';

    // Use pluggable adapter if provided (e.g. React Native native downloader)
    if (this.downloader) {
      await this.downloader.download(resolvedUrl, targetPath, options?.onProgress);
      const fileStat = await stat(targetPath);
      await this.cache.cacheModel(resolvedUrl, targetPath, fileStat.size);
      return targetPath;
    }

    let lastError: Error | undefined;

    for (let attempt = 0; attempt <= MAX_RETRIES; attempt++) {
      try {
        await downloadToFile(resolvedUrl, tmpPath, options);
        await rename(tmpPath, targetPath);

        const fileStat = await stat(targetPath);
        await this.cache.cacheModel(resolvedUrl, targetPath, fileStat.size);
        return targetPath;
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));

        try { await unlink(tmpPath); } catch { /* partial cleanup */ }

        if (attempt < MAX_RETRIES && isRetryableError(lastError)) {
          await sleep(BASE_DELAY_MS * Math.pow(2, attempt));
          continue;
        }
        throw lastError;
      }
    }

    throw lastError!;
  }

  /** List all cached models. */
  async listModels(): Promise<CacheEntry[]> {
    return this.cache.listModels();
  }

  /** Remove a cached model by its original URL. */
  async removeModel(url: string): Promise<boolean> {
    return this.cache.removeModel(resolveHuggingFaceUrl(url));
  }
}

// -- Download helpers --

async function downloadToFile(
  url: string,
  tmpPath: string,
  options?: DownloadOptions,
): Promise<void> {
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(`Download failed: ${response.status} ${response.statusText}`);
  }
  if (!response.body) {
    throw new Error('Download failed: no response body');
  }

  const totalSize = Number(response.headers.get('content-length') ?? 0);
  let downloaded = 0;

  const reader = response.body.getReader();
  const progressStream = new ReadableStream({
    async pull(controller) {
      const { done, value } = await reader.read();
      if (done) {
        controller.close();
        return;
      }
      downloaded += value.byteLength;
      if (options?.onProgress && totalSize > 0) {
        options.onProgress(downloaded, totalSize, (downloaded / totalSize) * 100);
      }
      controller.enqueue(value);
    },
  });

  const nodeStream = Readable.fromWeb(progressStream as import('node:stream/web').ReadableStream);
  const fileStream = createWriteStream(tmpPath);
  await pipeline(nodeStream, fileStream);

  if (totalSize > 0) {
    const fileStat = await stat(tmpPath);
    if (fileStat.size !== totalSize) {
      throw new Error(
        `Download incomplete: expected ${totalSize} bytes, got ${fileStat.size}`,
      );
    }
  }
}

function isRetryableError(err: Error): boolean {
  const msg = err.message;
  if (/Download failed: (5\d\d|429)/.test(msg)) return true;

  const lower = msg.toLowerCase();
  return lower.includes('econnreset') || lower.includes('econnrefused') ||
    lower.includes('etimedout') || lower.includes('epipe') ||
    lower.includes('socket hang up') || lower.includes('fetch failed') ||
    lower.includes('download incomplete');
}

function sleep(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// -- URL helpers --

/**
 * Resolve shorthand like "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama.gguf"
 * into a full HuggingFace download URL. Full URLs pass through unchanged.
 */
function resolveHuggingFaceUrl(input: string): string {
  // Already a full URL
  if (input.startsWith('http://') || input.startsWith('https://')) {
    return input;
  }

  // Shorthand: user/repo/file.gguf  or  user/repo/branch/file.gguf
  const parts = input.split('/');
  if (parts.length === 3) {
    const [user, repo, file] = parts;
    return `https://huggingface.co/${user}/${repo}/resolve/main/${file}`;
  }
  if (parts.length === 4) {
    const [user, repo, branch, file] = parts;
    return `https://huggingface.co/${user}/${repo}/resolve/${branch}/${file}`;
  }

  throw new Error(
    `Invalid model specifier: "${input}". Use a full URL or shorthand "user/repo/file.gguf".`,
  );
}

/** Extract the filename from a URL path. */
function extractFilename(url: string): string {
  const pathname = new URL(url).pathname;
  const segments = pathname.split('/');
  const filename = segments[segments.length - 1];
  return filename || 'model.gguf';
}
