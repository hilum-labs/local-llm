import { createHash } from 'node:crypto';
import { readFile, writeFile, mkdir, rm, stat, rename } from 'node:fs/promises';
import { join } from 'node:path';
import { homedir } from 'node:os';

export interface CacheEntry {
  url: string;
  path: string;
  size: number;
  downloadedAt: string;
  lastUsedAt: string;
}

interface CacheIndex {
  models: Record<string, CacheEntry>;
}

export class ModelCache {
  readonly cacheDir: string;
  private indexPath: string;

  constructor(cacheDir?: string) {
    this.cacheDir = cacheDir ?? join(homedir(), '.local-llm', 'models');
    this.indexPath = join(this.cacheDir, 'index.json');
  }

  /** Hash a URL to produce a stable directory name. */
  static hashUrl(url: string): string {
    return createHash('sha256').update(url).digest('hex').slice(0, 16);
  }

  /** Return the local path for a cached model, or null if not cached. */
  async getCachedModel(url: string): Promise<CacheEntry | null> {
    const index = await this.readIndex();
    const key = ModelCache.hashUrl(url);
    const entry = index.models[key];
    if (!entry) return null;

    // Verify file still exists on disk
    try {
      await stat(entry.path);
    } catch {
      // File was deleted externally — remove stale entry
      delete index.models[key];
      await this.writeIndex(index);
      return null;
    }

    // Update lastUsedAt
    entry.lastUsedAt = new Date().toISOString();
    await this.writeIndex(index);
    return entry;
  }

  /** Register a downloaded model in the cache index. */
  async cacheModel(url: string, filePath: string, size: number): Promise<CacheEntry> {
    const index = await this.readIndex();
    const key = ModelCache.hashUrl(url);
    const now = new Date().toISOString();
    const entry: CacheEntry = {
      url,
      path: filePath,
      size,
      downloadedAt: now,
      lastUsedAt: now,
    };
    index.models[key] = entry;
    await this.writeIndex(index);
    return entry;
  }

  /** List all cached models. */
  async listModels(): Promise<CacheEntry[]> {
    const index = await this.readIndex();
    return Object.values(index.models);
  }

  /** Remove a cached model by URL (deletes file + index entry). */
  async removeModel(url: string): Promise<boolean> {
    const index = await this.readIndex();
    const key = ModelCache.hashUrl(url);
    const entry = index.models[key];
    if (!entry) return false;

    // Delete the model's directory
    const modelDir = join(this.cacheDir, key);
    try {
      await rm(modelDir, { recursive: true, force: true });
    } catch {
      // best-effort
    }

    delete index.models[key];
    await this.writeIndex(index);
    return true;
  }

  /** Directory path for a given URL hash. */
  modelDir(url: string): string {
    return join(this.cacheDir, ModelCache.hashUrl(url));
  }

  // -- internal --

  private async readIndex(): Promise<CacheIndex> {
    let raw: string;
    try {
      raw = await readFile(this.indexPath, 'utf-8');
    } catch (err: unknown) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        return { models: {} };
      }
      throw err;
    }

    try {
      return JSON.parse(raw) as CacheIndex;
    } catch {
      const backupPath = this.indexPath + '.corrupt.' + Date.now();
      try { await rename(this.indexPath, backupPath); } catch { /* best-effort */ }
      console.warn(
        `[local-llm] Cache index was corrupt and has been reset. ` +
        `Backup saved to: ${backupPath}`,
      );
      return { models: {} };
    }
  }

  private async writeIndex(index: CacheIndex): Promise<void> {
    await mkdir(this.cacheDir, { recursive: true });
    const tmpPath = this.indexPath + '.tmp.' + process.pid;
    await writeFile(tmpPath, JSON.stringify(index, null, 2) + '\n');
    await rename(tmpPath, this.indexPath);
  }
}
