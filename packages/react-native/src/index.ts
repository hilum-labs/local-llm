// Override native loader before anything else imports it
import { setNativeAddon } from 'local-llm/native';
import { createReactNativeAddon } from './native-bridge';
setNativeAddon(createReactNativeAddon());

// Re-export everything from the shared package
export { LocalAI, Model, InferenceContext, EmbeddingContext, ModelManager, ModelPool, ChatCompletions, Embeddings } from 'local-llm';
export type {
  LocalAIOptions,
  ChatMessage, ContentPart, ComputeMode, GenerateOptions, FlashAttentionMode,
  KvCacheType, ContextOverflowStrategy, ContextOverflowConfig, EmbeddingPoolingType,
  ResponseFormat, ChatCompletionTool, ChatCompletionToolChoice,
  ModelOptions, ContextOptions, EmbeddingContextOptions, QuantizationType, QuantizeOptions,
  BatchResult, DownloadOptions,
} from 'local-llm';

// RN-specific exports
export { getDeviceCapabilities, canRunModel, recommendQuantization } from './device';
export type { DeviceCapabilities } from './device';
export { createNativeDownloader } from './rn-downloader';
export type { DownloadAdapter } from './download-adapter';
