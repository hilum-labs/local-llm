export type ComputeMode = 'cpu' | 'hybrid' | 'gpu' | 'auto';

export interface ModelOptions {
  /** Compute mode: 'auto' (default), 'cpu', 'hybrid', or 'gpu'. */
  compute?: ComputeMode;
  /** Override GPU layer count directly (advanced — prefer `compute`). */
  gpuLayers?: number;
  useMmap?: boolean;
}

export type FlashAttentionMode = 'auto' | 'enabled' | 'disabled';

/** KV cache data type. Lower precision saves memory at minimal quality cost. Requires flash attention. */
export type KvCacheType = 'f16' | 'q8_0' | 'q4_0';

export interface ContextOptions {
  contextSize?: number;
  batchSize?: number;
  threads?: number;
  /** Flash attention mode: 'auto' (default), 'enabled', 'disabled'. */
  flashAttention?: FlashAttentionMode;
  /** KV cache key type: 'f16' (default), 'q8_0', 'q4_0'. Requires flash attention. */
  kvCacheTypeK?: KvCacheType;
  /** KV cache value type: 'f16' (default), 'q8_0', 'q4_0'. Requires flash attention. */
  kvCacheTypeV?: KvCacheType;
  /** Maximum simultaneous sequences for batch inference. Default: 1. */
  maxSequences?: number;
}

/** Pooling strategy for embedding models. */
export type EmbeddingPoolingType = 'mean' | 'cls' | 'last';

export interface EmbeddingContextOptions {
  contextSize?: number;
  batchSize?: number;
  threads?: number;
  /** Pooling strategy for combining token embeddings. Default: 'mean'. */
  poolingType?: EmbeddingPoolingType;
}

export interface ResponseFormat {
  type: 'text' | 'json_object' | 'json_schema';
  /** JSON Schema object — required when type is 'json_schema'. */
  json_schema?: {
    name?: string;
    strict?: boolean;
    schema: Record<string, unknown>;
  };
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  /** Multiplicative penalty for repeated tokens (llama.cpp-specific, typically 1.0–1.3). */
  repeatPenalty?: number;
  /** Additive penalty proportional to token frequency (OpenAI-compatible, typically 0–2). */
  frequencyPenalty?: number;
  /** Additive penalty for any token that has appeared (OpenAI-compatible, typically 0–2). */
  presencePenalty?: number;
  seed?: number;
  /** Stop sequences — generation halts when any sequence is produced. */
  stop?: string[];
  /** AbortSignal to cancel generation in progress. */
  signal?: AbortSignal;
  /** GBNF grammar string to constrain model output. */
  grammar?: string;
  /** Root rule name for the GBNF grammar (default: "root"). */
  grammarRoot?: string;
  /** Structured output format — 'json_object' for free JSON, 'json_schema' for schema-constrained. */
  responseFormat?: ResponseFormat;
}

export interface TextContentPart {
  type: 'text';
  text: string;
}

export interface ImageContentPart {
  type: 'image_url';
  image_url: { url: string };
}

export type ContentPart = TextContentPart | ImageContentPart;

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | ContentPart[];
}

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type LogCallback = (level: LogLevel, message: string) => void;

/**
 * Strategy for handling conversations that exceed the model's context window.
 * - `'error'` — throw an error when messages exceed the context limit
 * - `'truncate_oldest'` — drop oldest non-system messages until the conversation fits
 * - `'sliding_window'` — keep system message + most recent messages that fit
 */
export type ContextOverflowStrategy = 'error' | 'truncate_oldest' | 'sliding_window';

/** Simplified message type used by the overflow callback. */
export interface OverflowMessage {
  role: string;
  content: string | Array<{ type: string; text?: string; image_url?: { url: string; detail?: string } }>;
}

/** Information passed to the onOverflow callback when the context budget is exceeded. */
export interface ContextOverflowEvent {
  /** The original conversation messages. */
  messages: OverflowMessage[];
  /** Number of messages in the original conversation. */
  originalMessageCount: number;
  /** Tokens required by the full formatted prompt. */
  promptTokens: number;
  /** Tokens available for the prompt (contextSize − reserved generation tokens). */
  availableTokens: number;
  /** Actual context window size in tokens. */
  contextSize: number;
  /** Tokens reserved for generation output. */
  reservedForGeneration: number;
  /** Strategy that will be applied if the callback doesn't return replacement messages. */
  strategy: ContextOverflowStrategy;
}

/**
 * Full configuration for context overflow management.
 * Pass a string (`ContextOverflowStrategy`) for simple usage,
 * or this object for fine-grained control.
 */
export interface ContextOverflowConfig {
  /** Strategy for handling overflow. Default: 'sliding_window'. */
  strategy?: ContextOverflowStrategy;
  /**
   * Fraction of the context window to reserve for generation when `max_tokens` is not set.
   * Default: 0.25 (25%). For example, a 4096-token context reserves 1024 tokens for output.
   */
  reserveRatio?: number;
  /**
   * Called when overflow is detected, before the strategy is applied.
   * Return replacement messages to override the default strategy (e.g. a summarized conversation).
   * Return `void` (or nothing) to let the configured strategy handle it.
   */
  onOverflow?: (event: ContextOverflowEvent) => OverflowMessage[] | void | Promise<OverflowMessage[] | void>;
}

// ── Quantization types ──────────────────────────────────────────────────────

export type QuantizationType =
  | 'F32' | 'F16' | 'BF16'
  | 'Q4_0' | 'Q4_1' | 'Q5_0' | 'Q5_1' | 'Q8_0'
  | 'Q2_K' | 'Q3_K_S' | 'Q3_K_M' | 'Q3_K_L'
  | 'Q4_K_S' | 'Q4_K_M' | 'Q5_K_S' | 'Q5_K_M' | 'Q6_K'
  | 'IQ2_XXS' | 'IQ2_XS' | 'IQ2_S' | 'IQ2_M'
  | 'IQ3_XXS' | 'IQ3_XS' | 'IQ3_S' | 'IQ3_M'
  | 'IQ4_NL' | 'IQ4_XS' | 'IQ1_S' | 'IQ1_M'
  | 'TQ1_0' | 'TQ2_0';

export interface QuantizeOptions {
  type: QuantizationType;
  threads?: number;
  allowRequantize?: boolean;
  quantizeOutputTensor?: boolean;
  pure?: boolean;
  dryRun?: boolean;
}

export interface BatchResult {
  index: number;
  text: string;
  finishReason: 'stop' | 'length';
}

/** Metadata about context window management attached to responses. */
export interface ContextMetadata {
  /** Whether context overflow management was triggered. */
  overflowTriggered: boolean;
  /** Strategy that was used (only set when overflow was triggered). */
  strategyUsed?: ContextOverflowStrategy;
  /** Number of messages in the original conversation. */
  originalMessageCount: number;
  /** Number of messages kept after truncation. */
  keptMessageCount: number;
  /** Actual context window size in tokens. */
  contextSize: number;
  /** Tokens used by the final prompt. */
  promptTokens: number;
  /** Tokens reserved for generation output. */
  reservedForGeneration: number;
}
