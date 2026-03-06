import type { TurboModule } from 'react-native';
import { TurboModuleRegistry } from 'react-native';

export interface Spec extends TurboModule {
  // ── Backend info ────────────────────────────────────────────────────────
  backendInfo(): string;
  backendVersion(): string;

  // ── Model lifecycle ─────────────────────────────────────────────────────
  loadModel(path: string, options: Object): Promise<string>;       // → model ID
  getModelSize(modelId: string): number;
  freeModel(modelId: string): void;

  // ── Context lifecycle ───────────────────────────────────────────────────
  createContext(modelId: string, options: Object): string;          // → context ID
  getContextSize(contextId: string): number;
  freeContext(contextId: string): void;

  // ── KV cache ────────────────────────────────────────────────────────────
  kvCacheClear(contextId: string, fromPos: number): void;

  // ── Tokenization ────────────────────────────────────────────────────────
  tokenize(modelId: string, text: string, addSpecial: boolean, parseSpecial: boolean): number[];
  detokenize(modelId: string, tokens: number[]): string;
  applyChatTemplate(modelId: string, messages: Object[], addAssistant: boolean): string;

  // ── Text inference ──────────────────────────────────────────────────────
  generate(modelId: string, contextId: string, prompt: string, options: Object): Promise<string>;
  startStream(modelId: string, contextId: string, prompt: string, options: Object): void;
  stopStream(contextId: string): void;

  // ── Vision ──────────────────────────────────────────────────────────────
  loadProjector(modelId: string, path: string, options: Object): string;  // → mtmd context ID
  supportVision(mtmdId: string): boolean;
  freeMtmdContext(mtmdId: string): void;
  generateVision(
    modelId: string, contextId: string, mtmdId: string,
    prompt: string, imageBase64s: string[], options: Object,
  ): Promise<string>;
  startStreamVision(
    modelId: string, contextId: string, mtmdId: string,
    prompt: string, imageBase64s: string[], options: Object,
  ): void;

  // ── Grammar / Structured output ─────────────────────────────────────────
  jsonSchemaToGrammar(schemaJson: string): string;

  // ── Embeddings ──────────────────────────────────────────────────────────
  getEmbeddingDimension(modelId: string): number;
  createEmbeddingContext(modelId: string, options: Object): string;  // → context ID
  embed(contextId: string, modelId: string, tokens: number[]): number[];
  embedBatch(contextId: string, modelId: string, tokenArrays: Object): Object;

  // ── Batch inference ─────────────────────────────────────────────────────
  startBatch(modelId: string, contextId: string, prompts: string[], options: Object): void;

  // ── Model quantization ──────────────────────────────────────────────────
  quantize(inputPath: string, outputPath: string, options: Object): void;

  // ── Logging ─────────────────────────────────────────────────────────────
  setLogLevel(level: number): void;
  enableLogEvents(enabled: boolean): void;

  // ── Downloads ───────────────────────────────────────────────────────────
  downloadModel(url: string, destPath: string): void;
  cancelDownload(url: string): void;

  // ── Device capabilities (RN-only) ──────────────────────────────────────
  getDeviceCapabilities(): Object;
  getModelStoragePath(): string;
}

export default TurboModuleRegistry.getEnforcing<Spec>('LocalLLM');
