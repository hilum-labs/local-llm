import { Model } from './engine.js';
import type { ModelOptions } from './types.js';

export interface ModelPoolOptions {
  /** Maximum total memory in bytes for all loaded models. 0 = unlimited. */
  maxMemoryBytes?: number;
  /** Maximum number of models to keep loaded. 0 = unlimited. */
  maxModels?: number;
}

interface PoolEntry {
  alias: string;
  model: Model;
  refCount: number;
  lastAccess: number;
  sizeBytes: number;
}

export interface PoolInfo {
  alias: string;
  sizeBytes: number;
  refCount: number;
  lastAccess: number;
}

export class ModelPool {
  private entries: Map<string, PoolEntry> = new Map();
  private maxMemoryBytes: number;
  private maxModels: number;

  constructor(options?: ModelPoolOptions) {
    this.maxMemoryBytes = options?.maxMemoryBytes ?? 0;
    this.maxModels = options?.maxModels ?? 0;
  }

  async load(alias: string, pathOrBuffer: string | Buffer, options?: ModelOptions): Promise<Model> {
    const existing = this.entries.get(alias);
    if (existing) {
      existing.refCount++;
      existing.lastAccess = Date.now();
      return existing.model;
    }

    // Evict if needed before loading
    this.evictIfNeeded();

    const model = typeof pathOrBuffer === 'string'
      ? new Model(pathOrBuffer, options)
      : Model.fromBuffer(pathOrBuffer, options);

    const sizeBytes = model.sizeBytes;

    // Evict further if the new model would exceed memory limits
    while (this.maxMemoryBytes > 0 && this.totalMemory() + sizeBytes > this.maxMemoryBytes) {
      if (!this.evictLRU()) break;
    }

    const entry: PoolEntry = {
      alias,
      model,
      refCount: 1,
      lastAccess: Date.now(),
      sizeBytes,
    };

    this.entries.set(alias, entry);
    return model;
  }

  get(alias: string): Model | undefined {
    const entry = this.entries.get(alias);
    if (entry) {
      entry.lastAccess = Date.now();
      return entry.model;
    }
    return undefined;
  }

  release(alias: string): boolean {
    const entry = this.entries.get(alias);
    if (!entry) return false;

    entry.refCount--;
    if (entry.refCount <= 0) {
      entry.model.dispose();
      this.entries.delete(alias);
    }
    return true;
  }

  unload(alias: string): void {
    const entry = this.entries.get(alias);
    if (!entry) return;

    entry.model.dispose();
    this.entries.delete(alias);
  }

  list(): PoolInfo[] {
    return Array.from(this.entries.values()).map(e => ({
      alias: e.alias,
      sizeBytes: e.sizeBytes,
      refCount: e.refCount,
      lastAccess: e.lastAccess,
    }));
  }

  dispose(): void {
    for (const entry of this.entries.values()) {
      entry.model.dispose();
    }
    this.entries.clear();
  }

  private totalMemory(): number {
    let total = 0;
    for (const entry of this.entries.values()) {
      total += entry.sizeBytes;
    }
    return total;
  }

  private evictIfNeeded(): void {
    // Evict by model count
    while (this.maxModels > 0 && this.entries.size >= this.maxModels) {
      if (!this.evictLRU()) break;
    }
  }

  /** Evict the least-recently-used model with refCount 0. Returns true if eviction happened. */
  private evictLRU(): boolean {
    let oldest: PoolEntry | null = null;
    for (const entry of this.entries.values()) {
      if (entry.refCount > 0) continue;
      if (!oldest || entry.lastAccess < oldest.lastAccess) {
        oldest = entry;
      }
    }
    if (!oldest) return false;

    oldest.model.dispose();
    this.entries.delete(oldest.alias);
    return true;
  }
}
