import type { Model, EmbeddingContext } from './engine.js';

// ── OpenAI-compatible types ──────────────────────────────────────────────────

export interface EmbeddingRequest {
  /** Text or array of texts to embed. */
  input: string | string[];
  /** Model identifier (ignored — model is set at construction time). */
  model?: string;
  /** Encoding format. Default: 'float'. */
  encoding_format?: 'float' | 'base64';
}

export interface EmbeddingData {
  object: 'embedding';
  index: number;
  embedding: number[] | string;
}

export interface EmbeddingUsage {
  prompt_tokens: number;
  total_tokens: number;
}

export interface EmbeddingResponse {
  object: 'list';
  model: string;
  data: EmbeddingData[];
  usage: EmbeddingUsage;
}

// ── Embeddings class ─────────────────────────────────────────────────────────

export class Embeddings {
  private model: Model;
  private context: EmbeddingContext;
  private modelName: string;

  constructor(model: Model, context: EmbeddingContext, modelName: string) {
    this.model = model;
    this.context = context;
    this.modelName = modelName;
  }

  async create(params: EmbeddingRequest): Promise<EmbeddingResponse> {
    const inputs = typeof params.input === 'string' ? [params.input] : params.input;
    const base64 = params.encoding_format === 'base64';

    let embeddings: Float32Array[];
    let totalTokens = 0;

    if (inputs.length === 1) {
      const tokens = this.model.tokenize(inputs[0]);
      totalTokens = tokens.length;
      embeddings = [this.context.embed(inputs[0])];
    } else {
      for (const text of inputs) {
        totalTokens += this.model.tokenize(text).length;
      }
      embeddings = this.context.embedBatch(inputs);
    }

    const data: EmbeddingData[] = embeddings.map((emb, i) => ({
      object: 'embedding' as const,
      index: i,
      embedding: base64
        ? Buffer.from(emb.buffer, emb.byteOffset, emb.byteLength).toString('base64')
        : Array.from(emb),
    }));

    return {
      object: 'list',
      model: params.model ?? this.modelName,
      data,
      usage: {
        prompt_tokens: totalTokens,
        total_tokens: totalTokens,
      },
    };
  }
}
