export { LocalLLM } from './hilum-ai.js';
export type { LocalLLMOptions } from './hilum-ai.js';
export { Model, InferenceContext, EmbeddingContext, optimalThreadCount, quantize } from './engine.js';
export { ChatCompletions } from './openai-compat.js';
export type { ChatCompletionsConfig } from './openai-compat.js';
export { Embeddings } from './embeddings.js';
export type { EmbeddingRequest, EmbeddingResponse, EmbeddingData, EmbeddingUsage } from './embeddings.js';
export { LocalLLMProvider } from './ai-sdk-provider.js';
export { setNativeAddon } from './native.js';
export type { NativeAddon } from './native.js';
export { ModelManager } from './model-manager.js';
export { ModelPool } from './model-pool.js';
export type { ModelPoolOptions, PoolInfo } from './model-pool.js';
export type { DownloadOptions } from './model-manager.js';
export type { DownloadAdapter } from './download-adapter.js';
export type { CacheEntry } from './cache.js';
export type { ModelOptions, ContextOptions, EmbeddingContextOptions, EmbeddingPoolingType, GenerateOptions, ChatMessage, ComputeMode, LogLevel, LogCallback, ContentPart, TextContentPart, ImageContentPart, ResponseFormat, ContextOverflowStrategy, ContextOverflowConfig, ContextOverflowEvent, ContextMetadata, OverflowMessage, FlashAttentionMode, KvCacheType, QuantizationType, QuantizeOptions, BatchResult, InferenceMetrics, BenchmarkOptions, BenchmarkResult } from './types.js';
export type {
  ChatCompletionRequest,
  ChatCompletionRequestMessage,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ChatCompletionUsage,
  ChatCompletionMessage,
  ChatCompletionChoice,
  ChatCompletionChunkChoice,
  ChatCompletionChunkDelta,
  ChatCompletionResponseFormat,
  ChatCompletionTool,
  ChatCompletionToolChoice,
  ChatCompletionToolCall,
  ChatCompletionToolCallDelta,
} from './openai-types.js';
