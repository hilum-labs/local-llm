import type { Model, InferenceContext } from './engine.js';
import { resolveImageToBuffer } from './engine.js';
import type { GenerateOptions, ChatMessage } from './types.js';

// LanguageModelV3 types — imported dynamically to keep 'ai' an optional peer dep.
// We inline the subset of types we need so the module compiles without 'ai' installed.

interface LanguageModelV3Message {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: unknown;
}

type LanguageModelV3Prompt = LanguageModelV3Message[];

interface LanguageModelV3CallOptions {
  prompt: LanguageModelV3Prompt;
  maxOutputTokens?: number;
  temperature?: number;
  stopSequences?: string[];
  topP?: number;
  topK?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  responseFormat?: { type: string };
  seed?: number;
  tools?: unknown[];
  toolChoice?: unknown;
  abortSignal?: AbortSignal;
  [key: string]: unknown;
}

interface LanguageModelV3Usage {
  inputTokens: number;
  outputTokens: number;
}

type LanguageModelV3FinishReason = 'stop' | 'length' | 'content-filter' | 'tool-calls' | 'error' | 'other' | 'unknown';

interface LanguageModelV3Content {
  type: 'text';
  text: string;
}

interface LanguageModelV3GenerateResult {
  content: LanguageModelV3Content[];
  finishReason: LanguageModelV3FinishReason;
  usage: LanguageModelV3Usage;
  warnings: Array<{ type: string; message: string }>;
  response?: unknown;
  request?: unknown;
  providerMetadata?: unknown;
}

interface LanguageModelV3StreamPart {
  type: string;
  [key: string]: unknown;
}

interface LanguageModelV3StreamResult {
  stream: ReadableStream<LanguageModelV3StreamPart>;
  request?: unknown;
  response?: unknown;
}

interface LanguageModelV3 {
  readonly specificationVersion: 'v3';
  readonly provider: string;
  readonly modelId: string;
  supportedUrls: Record<string, RegExp[]>;
  doGenerate(options: LanguageModelV3CallOptions): PromiseLike<LanguageModelV3GenerateResult>;
  doStream(options: LanguageModelV3CallOptions): PromiseLike<LanguageModelV3StreamResult>;
}

/**
 * Extract plain text from a LanguageModelV3 message content.
 * Content can be a string, or an array of typed parts.
 */
function extractText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((part: { type: string }) => part.type === 'text')
      .map((part: { text: string }) => part.text)
      .join('');
  }
  return String(content ?? '');
}

/** Check if a LanguageModelV3Prompt contains image parts. */
function promptHasImages(prompt: LanguageModelV3Prompt): boolean {
  return prompt.some((msg) => {
    if (!Array.isArray(msg.content)) return false;
    return (msg.content as Array<{ type: string }>).some((p) => p.type === 'image');
  });
}

/**
 * Convert LanguageModelV3Prompt to our ChatMessage format for applyChatTemplate.
 */
function promptToChatMessages(prompt: LanguageModelV3Prompt): ChatMessage[] {
  const messages: ChatMessage[] = [];

  for (const msg of prompt) {
    if (msg.role === 'tool') {
      messages.push({ role: 'user', content: extractText(msg.content) });
      continue;
    }

    const role = msg.role as 'system' | 'user' | 'assistant';

    if (Array.isArray(msg.content) && (msg.content as Array<{ type: string }>).some((p) => p.type === 'image')) {
      // Build multimodal content parts
      const parts = (msg.content as Array<{ type: string; text?: string; image?: Uint8Array; mimeType?: string }>).map((p) => {
        if (p.type === 'text') {
          return { type: 'text' as const, text: p.text ?? '' };
        }
        if (p.type === 'image' && p.image) {
          const base64 = Buffer.from(p.image).toString('base64');
          const mime = p.mimeType ?? 'image/png';
          return {
            type: 'image_url' as const,
            image_url: { url: `data:${mime};base64,${base64}` },
          };
        }
        // Skip unknown parts
        return { type: 'text' as const, text: '' };
      }).filter((p) => p.type === 'image_url' || (p.type === 'text' && p.text));

      messages.push({ role, content: parts });
    } else {
      const text = extractText(msg.content);
      if (text) {
        messages.push({ role, content: text });
      }
    }
  }

  return messages;
}

/**
 * Map LanguageModelV3CallOptions to our GenerateOptions.
 */
function mapOptions(options: LanguageModelV3CallOptions): GenerateOptions {
  const genOpts: GenerateOptions = {
    maxTokens: options.maxOutputTokens,
    temperature: options.temperature,
    topP: options.topP,
    topK: options.topK,
    frequencyPenalty: options.frequencyPenalty as number | undefined,
    presencePenalty: options.presencePenalty as number | undefined,
    seed: options.seed,
    stop: options.stopSequences,
    signal: options.abortSignal,
  };

  if (options.responseFormat) {
    const rfType = options.responseFormat.type;
    if (rfType === 'json') {
      genOpts.responseFormat = { type: 'json_object' };
    } else if (rfType === 'json-schema' && 'schema' in options.responseFormat) {
      genOpts.responseFormat = {
        type: 'json_schema',
        json_schema: {
          schema: (options.responseFormat as { schema: Record<string, unknown> }).schema,
        },
      };
    }
  }

  return genOpts;
}

/**
 * Collect unsupported option warnings.
 */
function collectWarnings(options: LanguageModelV3CallOptions): Array<{ type: string; message: string }> {
  const warnings: Array<{ type: string; message: string }> = [];

  if (options.tools && options.tools.length > 0) {
    warnings.push({ type: 'unsupported-setting', message: 'Tool use is not supported via AI SDK provider. Use ai.chat.completions.create() with tools directly.' });
  }

  return warnings;
}

/**
 * Vercel AI SDK LanguageModelV3 provider backed by a local llama.cpp model.
 */
export class LocalLLMProvider implements LanguageModelV3 {
  readonly specificationVersion = 'v3' as const;
  readonly provider = 'local-llm';
  readonly modelId: string;
  readonly supportedUrls: Record<string, RegExp[]> = {};

  private model: Model;
  private context: InferenceContext;

  constructor(model: Model, context: InferenceContext, modelId: string) {
    this.model = model;
    this.context = context;
    this.modelId = modelId;
  }

  async doGenerate(options: LanguageModelV3CallOptions): Promise<LanguageModelV3GenerateResult> {
    const messages = promptToChatMessages(options.prompt);
    const genOpts = mapOptions(options);
    const isVision = promptHasImages(options.prompt);

    let text: string;
    let inputTokens: number;

    if (isVision) {
      const { buildVisionPrompt } = await import('./engine.js');
      const { text: visionText, imageBuffers } = await buildVisionPrompt(messages);
      const prompt = this.model.applyChatTemplate(
        [{ role: 'user', content: visionText }], true,
      );
      inputTokens = this.model.tokenize(prompt).length;
      text = await this.context.generateVision(prompt, imageBuffers, genOpts);
    } else {
      const prompt = this.model.applyChatTemplate(
        messages.map((m) => ({ ...m, content: typeof m.content === 'string' ? m.content : '' })), true,
      );
      inputTokens = this.model.tokenize(prompt).length;
      text = await this.context.generate(prompt, genOpts);
    }

    const outputTokens = this.model.tokenize(text).length;
    const maxTokens = options.maxOutputTokens ?? Infinity;
    const finishReason: LanguageModelV3FinishReason = outputTokens >= maxTokens ? 'length' : 'stop';

    return {
      content: [{ type: 'text', text }],
      finishReason,
      usage: { inputTokens, outputTokens },
      warnings: collectWarnings(options),
    };
  }

  async doStream(options: LanguageModelV3CallOptions): Promise<LanguageModelV3StreamResult> {
    const messages = promptToChatMessages(options.prompt);
    const genOpts = mapOptions(options);
    const warnings = collectWarnings(options);
    const isVision = promptHasImages(options.prompt);
    const model = this.model;
    const context = this.context;
    const maxTokens = options.maxOutputTokens ?? Infinity;

    let prompt: string;
    let tokenStream: AsyncGenerator<string>;
    let inputTokens: number;

    if (isVision) {
      const { buildVisionPrompt } = await import('./engine.js');
      const { text: visionText, imageBuffers } = await buildVisionPrompt(messages);
      prompt = model.applyChatTemplate([{ role: 'user', content: visionText }], true);
      inputTokens = model.tokenize(prompt).length;
      tokenStream = context.streamVision(prompt, imageBuffers, genOpts);
    } else {
      prompt = model.applyChatTemplate(
        messages.map((m) => ({ ...m, content: typeof m.content === 'string' ? m.content : '' })), true,
      );
      inputTokens = model.tokenize(prompt).length;
      tokenStream = context.stream(prompt, genOpts);
    }

    const stream = new ReadableStream<LanguageModelV3StreamPart>({
      async start(controller) {
        controller.enqueue({ type: 'stream-start', warnings });

        let outputTokens = 0;
        let fullText = '';

        try {
          controller.enqueue({ type: 'text-start', id: '0' });

          for await (const token of tokenStream) {
            fullText += token;
            controller.enqueue({ type: 'text-delta', textDelta: token });
          }

          controller.enqueue({ type: 'text-end' });

          outputTokens = model.tokenize(fullText).length;

          const finishReason: LanguageModelV3FinishReason = outputTokens >= maxTokens ? 'length' : 'stop';

          controller.enqueue({
            type: 'finish',
            finishReason,
            usage: { inputTokens, outputTokens },
          });
        } catch (err) {
          controller.enqueue({
            type: 'error',
            error: err instanceof Error ? err : new Error(String(err)),
          });
        } finally {
          controller.close();
        }
      },
    });

    return { stream };
  }
}
