import { randomUUID } from 'node:crypto';
import type { Model, InferenceContext } from './engine.js';
import { hasImageContent, buildVisionPrompt } from './engine.js';
import type {
  ChatMessage,
  GenerateOptions,
  BatchResult,
  ContextOverflowStrategy,
  ContextOverflowConfig,
  ContextOverflowEvent,
  ContextMetadata,
  OverflowMessage,
  InferenceMetrics,
} from './types.js';
import type {
  ChatCompletionRequest,
  ChatCompletionRequestMessage,
  ChatCompletionResponse,
  ChatCompletionChunk,
  ChatCompletionTool,
  ChatCompletionToolChoice,
  ChatCompletionToolCall,
} from './openai-types.js';

// ── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_STRATEGY: ContextOverflowStrategy = 'sliding_window';
const DEFAULT_RESERVE_RATIO = 0.25;

// ── Config ───────────────────────────────────────────────────────────────────

export interface ChatCompletionsConfig {
  contextOverflow?: ContextOverflowStrategy | ContextOverflowConfig;
}

interface NormalizedOverflowConfig {
  strategy: ContextOverflowStrategy;
  reserveRatio: number;
  onOverflow?: ContextOverflowConfig['onOverflow'];
}

function normalizeOverflowConfig(
  input?: ContextOverflowStrategy | ContextOverflowConfig,
): NormalizedOverflowConfig {
  if (!input) {
    return { strategy: DEFAULT_STRATEGY, reserveRatio: DEFAULT_RESERVE_RATIO };
  }
  if (typeof input === 'string') {
    return { strategy: input, reserveRatio: DEFAULT_RESERVE_RATIO };
  }
  return {
    strategy: input.strategy ?? DEFAULT_STRATEGY,
    reserveRatio: input.reserveRatio ?? DEFAULT_RESERVE_RATIO,
    onOverflow: input.onOverflow,
  };
}

// ── Tool context ─────────────────────────────────────────────────────────────

interface ToolContext {
  tools: ChatCompletionTool[];
  toolChoice: ChatCompletionToolChoice;
  grammarConstrained: boolean;
}

// ── ChatCompletions ──────────────────────────────────────────────────────────

export class ChatCompletions {
  private model: Model;
  private context: InferenceContext;
  private modelName: string;
  private overflowConfig: NormalizedOverflowConfig;
  /** Cached prompt template result to avoid redundant applyChatTemplate calls. */
  private cachedTemplateKey: string | null = null;
  private cachedTemplateResult: string | null = null;
  /** Tracks system prompt to guarantee KV cache prefix stability across turns. */
  private lastSystemPrompt: string | null = null;

  constructor(
    model: Model,
    context: InferenceContext,
    modelName: string,
    config?: ChatCompletionsConfig,
  ) {
    this.model = model;
    this.context = context;
    this.modelName = modelName;
    this.overflowConfig = normalizeOverflowConfig(config?.contextOverflow);
  }

  async create(
    params: ChatCompletionRequest & { stream: true },
  ): Promise<AsyncIterable<ChatCompletionChunk>>;
  async create(
    params: ChatCompletionRequest & { stream?: false },
  ): Promise<ChatCompletionResponse>;
  async create(
    params: ChatCompletionRequest,
  ): Promise<ChatCompletionResponse | AsyncIterable<ChatCompletionChunk>>;
  async create(
    params: ChatCompletionRequest,
  ): Promise<ChatCompletionResponse | AsyncIterable<ChatCompletionChunk>> {
    const genOpts: GenerateOptions = {
      maxTokens: params.max_tokens,
      temperature: params.temperature,
      topP: params.top_p,
      topK: params.top_k,
      seed: params.seed,
      frequencyPenalty: params.frequency_penalty,
      presencePenalty: params.presence_penalty,
      stop: params.stop,
      grammar: params.grammar,
      responseFormat: params.response_format ? {
        type: params.response_format.type,
        json_schema: params.response_format.json_schema,
      } : undefined,
    };

    // ── Tool calling setup ────────────────────────────────────────────────

    let toolCtx: ToolContext | undefined;
    let messages = params.messages;

    const hasTools = params.tools && params.tools.length > 0;
    if (hasTools) {
      const toolChoice = params.tool_choice ?? 'auto';
      const grammarConstrained = toolChoice === 'required' || typeof toolChoice === 'object';

      toolCtx = { tools: params.tools!, toolChoice, grammarConstrained };

      if (toolChoice !== 'none') {
        messages = prepareToolMessages(messages, params.tools!);

        if (grammarConstrained) {
          const schema = buildToolCallSchema(params.tools!, toolChoice);
          genOpts.responseFormat = {
            type: 'json_schema',
            json_schema: { name: 'tool_call', schema },
          };
          genOpts.grammar = undefined;
        }
      } else {
        toolCtx = undefined;
      }
    }

    // ── System prompt stability ─────────────────────────────────────────────
    // Track the system prompt so KV cache prefix is maximally reused.
    // When the system prompt changes between calls, invalidate the template
    // cache so the new prefix propagates correctly to prepareKvCache.

    const currentSystemPrompt = messages
      .filter((m) => m.role === 'system')
      .map((m) => typeof m.content === 'string' ? m.content : '')
      .join('\n');

    if (this.lastSystemPrompt !== null && currentSystemPrompt !== this.lastSystemPrompt) {
      // System prompt changed — invalidate template cache to force new formatting
      this.cachedTemplateKey = null;
      this.cachedTemplateResult = null;
    }
    this.lastSystemPrompt = currentSystemPrompt;

    // ── Context window management ──────────────────────────────────────────

    const contextSize = this.context.contextSize;
    const activeStrategy = params.context_overflow ?? this.overflowConfig.strategy;
    const reserveRatio = this.overflowConfig.reserveRatio;

    const originalMessageCount = messages.length;
    const promptTokens = this.countPromptTokens(messages);

    // When max_tokens is not specified, use remaining context after the prompt
    // (up to 75% of context) instead of a fixed reserve ratio.
    const reservedForGeneration = params.max_tokens
      ?? Math.max(1, Math.min(contextSize - promptTokens - 64, Math.floor(contextSize * 0.75)));
    const availableForPrompt = contextSize - reservedForGeneration;

    if (genOpts.maxTokens === undefined) {
      genOpts.maxTokens = Math.max(1, reservedForGeneration);
    }

    let contextMeta: ContextMetadata = {
      overflowTriggered: false,
      originalMessageCount,
      keptMessageCount: originalMessageCount,
      contextSize,
      promptTokens,
      reservedForGeneration,
    };

    if (promptTokens > availableForPrompt) {
      if (this.overflowConfig.onOverflow) {
        const event: ContextOverflowEvent = {
          messages: messages as OverflowMessage[],
          originalMessageCount,
          promptTokens,
          availableTokens: availableForPrompt,
          contextSize,
          reservedForGeneration,
          strategy: activeStrategy,
        };

        const replacement = await this.overflowConfig.onOverflow(event);
        if (replacement && replacement.length > 0) {
          messages = replacement as typeof messages;
          const newTokens = this.countPromptTokens(messages);
          contextMeta = {
            overflowTriggered: true,
            strategyUsed: 'sliding_window',
            originalMessageCount,
            keptMessageCount: messages.length,
            contextSize,
            promptTokens: newTokens,
            reservedForGeneration,
          };

          if (newTokens <= availableForPrompt) {
            return this.dispatch(params, messages, genOpts, contextMeta, toolCtx);
          }
        }
      }

      messages = this.trimMessages(messages, availableForPrompt, activeStrategy);
      const finalTokens = this.countPromptTokens(messages);

      contextMeta = {
        overflowTriggered: true,
        strategyUsed: activeStrategy,
        originalMessageCount,
        keptMessageCount: messages.length,
        contextSize,
        promptTokens: finalTokens,
        reservedForGeneration,
      };
    }

    return this.dispatch(params, messages, genOpts, contextMeta, toolCtx);
  }

  async createBatch(paramsList: ChatCompletionRequest[]): Promise<ChatCompletionResponse[]> {
    const prompts: string[] = [];
    const genOptsList: GenerateOptions[] = [];

    for (const params of paramsList) {
      const prompt = this.formatPrompt(params.messages);
      prompts.push(prompt);

      genOptsList.push({
        maxTokens: params.max_tokens,
        temperature: params.temperature,
        topP: params.top_p,
        topK: params.top_k,
        seed: params.seed,
        frequencyPenalty: params.frequency_penalty,
        presencePenalty: params.presence_penalty,
        stop: params.stop,
        grammar: params.grammar,
        responseFormat: params.response_format ? {
          type: params.response_format.type,
          json_schema: params.response_format.json_schema,
        } : undefined,
      });
    }

    const batchResults = await this.context.generateBatch(prompts, genOptsList);

    return batchResults.map((result, i) => {
      const params = paramsList[i];
      const promptTokens = this.model.tokenize(prompts[i]).length;
      const completionTokens = this.model.tokenize(result.text).length;
      const model = params.model ?? this.modelName;

      return {
        id: `chatcmpl-${randomUUID()}`,
        object: 'chat.completion' as const,
        created: Math.floor(Date.now() / 1000),
        model,
        choices: [{
          index: 0,
          message: { role: 'assistant' as const, content: result.text },
          finish_reason: result.finishReason,
        }],
        usage: {
          prompt_tokens: promptTokens,
          completion_tokens: completionTokens,
          total_tokens: promptTokens + completionTokens,
        },
      };
    });
  }

  // ── Prompt formatting & token counting ───────────────────────────────────

  private formatPrompt(messages: ChatCompletionRequestMessage[]): string {
    const key = JSON.stringify(messages);
    if (key === this.cachedTemplateKey && this.cachedTemplateResult) {
      return this.cachedTemplateResult;
    }
    const chatMessages = messagesToChat(messages);
    const result = this.model.applyChatTemplate(chatMessages, true);
    this.cachedTemplateKey = key;
    this.cachedTemplateResult = result;
    return result;
  }

  private countPromptTokens(
    messages: ChatCompletionRequestMessage[],
  ): number {
    return this.model.tokenize(this.formatPrompt(messages)).length;
  }

  // ── Context overflow strategies ──────────────────────────────────────────

  private trimMessages(
    messages: ChatCompletionRequestMessage[],
    availableTokens: number,
    strategy: ContextOverflowStrategy,
  ): ChatCompletionRequestMessage[] {
    if (strategy === 'error') {
      const promptTokens = this.countPromptTokens(messages);
      throw new Error(
        `Context window overflow: prompt requires ~${promptTokens} tokens ` +
        `but only ${availableTokens} are available (context ${this.context.contextSize} ` +
        `minus generation reserve). ` +
        `Reduce message count or set contextOverflow to 'truncate_oldest' or 'sliding_window'.`,
      );
    }

    if (strategy === 'truncate_oldest') {
      return this.truncateOldest(messages, availableTokens);
    }

    return this.slidingWindow(messages, availableTokens);
  }

  private truncateOldest(
    messages: ChatCompletionRequestMessage[],
    availableTokens: number,
  ): ChatCompletionRequestMessage[] {
    const result = [...messages];
    let firstNonSystem = 0;
    while (firstNonSystem < result.length && result[firstNonSystem].role === 'system') {
      firstNonSystem++;
    }

    while (firstNonSystem < result.length) {
      const tokens = this.countPromptTokens(result);
      if (tokens <= availableTokens) break;
      result.splice(firstNonSystem, 1);
    }

    return result;
  }

  private slidingWindow(
    messages: ChatCompletionRequestMessage[],
    availableTokens: number,
  ): ChatCompletionRequestMessage[] {
    const systemMessages: ChatCompletionRequestMessage[] = [];
    const nonSystemMessages: ChatCompletionRequestMessage[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemMessages.push(msg);
      } else {
        nonSystemMessages.push(msg);
      }
    }

    if (nonSystemMessages.length === 0) return systemMessages;

    let lo = 0;
    let hi = nonSystemMessages.length;

    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      const candidate = [...systemMessages, ...nonSystemMessages.slice(nonSystemMessages.length - mid)];
      const tokens = this.countPromptTokens(candidate);
      if (tokens <= availableTokens) {
        lo = mid;
      } else {
        hi = mid - 1;
      }
    }

    const kept = lo;
    if (kept === 0) return systemMessages;
    return [...systemMessages, ...nonSystemMessages.slice(nonSystemMessages.length - kept)];
  }

  // ── Dispatch to inference ────────────────────────────────────────────────

  private async dispatch(
    params: ChatCompletionRequest,
    messages: ChatCompletionRequestMessage[],
    genOpts: GenerateOptions,
    contextMeta: ContextMetadata,
    toolCtx?: ToolContext,
  ): Promise<ChatCompletionResponse | AsyncIterable<ChatCompletionChunk>> {
    let prompt: string;
    let imageBuffers: Buffer[] | undefined;

    if (hasImageContent(messages)) {
      const chatMessages = messages.map((m) => ({
        role: m.role as 'system' | 'user' | 'assistant',
        content: m.content,
      })) as ChatMessage[];

      const vision = await buildVisionPrompt(chatMessages);
      prompt = this.model.applyChatTemplate(
        [{ role: 'user' as const, content: vision.text }],
        true,
      );
      imageBuffers = vision.imageBuffers;
    } else {
      // Reuse cached prompt from countPromptTokens when available
      prompt = this.formatPrompt(messages);
    }

    if (params.stream) {
      return this.createStream(prompt, genOpts, contextMeta, toolCtx, imageBuffers);
    }
    return this.createNonStream(prompt, genOpts, params, contextMeta, toolCtx, imageBuffers);
  }

  // ── Non-streaming ────────────────────────────────────────────────────────

  private async createNonStream(
    prompt: string,
    genOpts: GenerateOptions,
    params: ChatCompletionRequest,
    contextMeta: ContextMetadata,
    toolCtx?: ToolContext,
    imageBuffers?: Buffer[],
  ): Promise<ChatCompletionResponse> {
    const promptTokens = this.model.tokenize(prompt).length;
    const content = imageBuffers
      ? await this.context.generateVision(prompt, imageBuffers, genOpts)
      : await this.context.generate(prompt, genOpts);
    const completionTokens = this.model.tokenize(content).length;
    const timing = this.context.getPerf() ?? undefined;

    contextMeta.promptTokens = promptTokens;

    const id = `chatcmpl-${randomUUID()}`;
    const created = Math.floor(Date.now() / 1000);
    const model = params.model ?? this.modelName;
    const usage = {
      prompt_tokens: promptTokens,
      completion_tokens: completionTokens,
      total_tokens: promptTokens + completionTokens,
    };

    if (toolCtx && toolCtx.toolChoice !== 'none') {
      const toolCalls = parseToolCalls(content, toolCtx.tools);
      if (toolCalls) {
        return {
          id, object: 'chat.completion', created, model,
          choices: [{
            index: 0,
            message: { role: 'assistant', content: null, tool_calls: toolCalls },
            finish_reason: 'tool_calls',
          }],
          usage,
          _context: contextMeta,
          _timing: timing,
        };
      }
    }

    return {
      id, object: 'chat.completion', created, model,
      choices: [{
        index: 0,
        message: { role: 'assistant', content },
        finish_reason: completionTokens >= (params.max_tokens ?? Infinity) ? 'length' : 'stop',
      }],
      usage,
      _context: contextMeta,
      _timing: timing,
    };
  }

  // ── Streaming ────────────────────────────────────────────────────────────

  private async *createStream(
    prompt: string,
    genOpts: GenerateOptions,
    contextMeta: ContextMetadata,
    toolCtx?: ToolContext,
    imageBuffers?: Buffer[],
  ): AsyncGenerator<ChatCompletionChunk> {
    const id = `chatcmpl-${randomUUID()}`;
    const created = Math.floor(Date.now() / 1000);
    const model = this.modelName;

    contextMeta.promptTokens = this.model.tokenize(prompt).length;

    const tokenStream = imageBuffers
      ? this.context.streamVision(prompt, imageBuffers, genOpts)
      : this.context.stream(prompt, genOpts);

    // When tools are active, buffer the full response to determine if it's a
    // tool call or regular text, then emit the appropriate delta format.
    if (toolCtx && toolCtx.toolChoice !== 'none') {
      const tokens: string[] = [];
      for await (const token of tokenStream) {
        tokens.push(token);
      }

      const fullContent = tokens.join('');
      const toolCalls = parseToolCalls(fullContent, toolCtx.tools);
      const toolTiming = this.context.getPerf() ?? undefined;

      if (toolCalls) {
        yield* this.emitToolCallChunks(id, created, model, contextMeta, toolCalls, toolTiming);
      } else {
        yield* this.emitTextChunks(id, created, model, contextMeta, tokens, toolTiming);
      }
      return;
    }

    // No tools — stream text tokens directly
    yield {
      id, object: 'chat.completion.chunk', created, model,
      choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
      _context: contextMeta,
    };

    for await (const token of tokenStream) {
      yield {
        id, object: 'chat.completion.chunk', created, model,
        choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
      };
    }

    yield {
      id, object: 'chat.completion.chunk', created, model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      _timing: this.context.getPerf() ?? undefined,
    };
  }

  /**
   * Emit OpenAI-compatible streaming chunks for tool calls.
   * Format: initial chunk with function name → arguments chunk → finish chunk.
   */
  private *emitToolCallChunks(
    id: string,
    created: number,
    model: string,
    contextMeta: ContextMetadata,
    toolCalls: ChatCompletionToolCall[],
    timing?: InferenceMetrics,
  ): Generator<ChatCompletionChunk> {
    // First chunk: role + tool call IDs and function names
    yield {
      id, object: 'chat.completion.chunk', created, model,
      choices: [{
        index: 0,
        delta: {
          role: 'assistant',
          tool_calls: toolCalls.map((tc, i) => ({
            index: i,
            id: tc.id,
            type: 'function' as const,
            function: { name: tc.function.name, arguments: '' },
          })),
        },
        finish_reason: null,
      }],
      _context: contextMeta,
    };

    // Arguments chunks — one per tool call with the full arguments
    for (let i = 0; i < toolCalls.length; i++) {
      yield {
        id, object: 'chat.completion.chunk', created, model,
        choices: [{
          index: 0,
          delta: {
            tool_calls: [{
              index: i,
              function: { arguments: toolCalls[i].function.arguments },
            }],
          },
          finish_reason: null,
        }],
      };
    }

    // Final chunk
    yield {
      id, object: 'chat.completion.chunk', created, model,
      choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }],
      _timing: timing,
    };
  }

  /** Replay buffered text tokens as regular streaming chunks. */
  private *emitTextChunks(
    id: string,
    created: number,
    model: string,
    contextMeta: ContextMetadata,
    tokens: string[],
    timing?: InferenceMetrics,
  ): Generator<ChatCompletionChunk> {
    yield {
      id, object: 'chat.completion.chunk', created, model,
      choices: [{ index: 0, delta: { role: 'assistant' }, finish_reason: null }],
      _context: contextMeta,
    };

    for (const token of tokens) {
      yield {
        id, object: 'chat.completion.chunk', created, model,
        choices: [{ index: 0, delta: { content: token }, finish_reason: null }],
      };
    }

    yield {
      id, object: 'chat.completion.chunk', created, model,
      choices: [{ index: 0, delta: {}, finish_reason: 'stop' }],
      _timing: timing,
    };
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function messagesToChat(
  messages: ChatCompletionRequestMessage[],
): Array<{ role: 'system' | 'user' | 'assistant'; content: string }> {
  return messages.map((m) => ({
    role: m.role as 'system' | 'user' | 'assistant',
    content: typeof m.content === 'string'
      ? m.content
      : m.content == null
        ? ''
        : extractTextFromParts(m.content),
  }));
}

function extractTextFromParts(
  parts: Array<{ type: string; text?: string }>,
): string {
  return parts
    .filter((p) => p.type === 'text' && p.text)
    .map((p) => p.text!)
    .join('');
}

// ── Tool calling helpers ─────────────────────────────────────────────────────

/**
 * Inject tool descriptions into the system prompt and convert tool/assistant
 * messages with tool_calls into a format compatible with any chat template.
 */
function prepareToolMessages(
  messages: ChatCompletionRequestMessage[],
  tools: ChatCompletionTool[],
): ChatCompletionRequestMessage[] {
  const toolPrompt = buildToolPrompt(tools);

  const result = messages.map((msg): ChatCompletionRequestMessage => {
    // Convert 'tool' role → 'user' for model-agnostic compatibility
    if (msg.role === 'tool') {
      const fnName = msg.name ?? 'unknown';
      return {
        role: 'user',
        content: `[Tool result for ${fnName}]: ${typeof msg.content === 'string' ? msg.content : JSON.stringify(msg.content)}`,
      };
    }

    // Convert assistant messages that contain tool_calls → assistant text
    if (msg.role === 'assistant' && msg.tool_calls && msg.tool_calls.length > 0) {
      const callText = msg.tool_calls.map((tc) => {
        let args: string;
        try {
          args = JSON.stringify(JSON.parse(tc.function.arguments));
        } catch {
          args = tc.function.arguments;
        }
        return JSON.stringify({ name: tc.function.name, arguments: JSON.parse(args) });
      }).join('\n');
      return { role: 'assistant', content: callText };
    }

    return msg;
  });

  // Inject tool prompt into the system message, or prepend one
  const systemIdx = result.findIndex((m) => m.role === 'system');
  if (systemIdx >= 0) {
    const existing = typeof result[systemIdx].content === 'string'
      ? result[systemIdx].content as string
      : '';
    result[systemIdx] = { ...result[systemIdx], content: existing + '\n\n' + toolPrompt };
  } else {
    result.unshift({ role: 'system', content: toolPrompt });
  }

  return result;
}

/**
 * Build a clear, model-agnostic tool prompt that describes available functions.
 */
function buildToolPrompt(tools: ChatCompletionTool[]): string {
  const descriptions = tools.map((t) => {
    const fn = t.function;
    const lines: string[] = [`### ${fn.name}`];
    if (fn.description) lines.push(fn.description);
    if (fn.parameters) lines.push(`Parameters: ${JSON.stringify(fn.parameters)}`);
    return lines.join('\n');
  }).join('\n\n');

  return (
    'You have access to the following tools:\n\n' +
    descriptions + '\n\n' +
    'To call a tool, respond with ONLY a JSON object in this exact format (no other text):\n' +
    '{"name": "tool_name", "arguments": {...}}\n\n' +
    'If you do not need to call a tool, respond normally with text.'
  );
}

/**
 * Build a JSON Schema that constrains model output to a valid tool call.
 * Used with `jsonSchemaToGrammar` for grammar-constrained modes.
 */
function buildToolCallSchema(
  tools: ChatCompletionTool[],
  toolChoice: 'required' | { type: 'function'; function: { name: string } },
): Record<string, unknown> {
  const buildSingleToolSchema = (tool: ChatCompletionTool): Record<string, unknown> => ({
    type: 'object',
    properties: {
      name: { const: tool.function.name },
      arguments: tool.function.parameters ?? { type: 'object' },
    },
    required: ['name', 'arguments'],
    additionalProperties: false,
  });

  if (typeof toolChoice === 'object') {
    const tool = tools.find((t) => t.function.name === toolChoice.function.name);
    if (!tool) {
      throw new Error(
        `Tool "${toolChoice.function.name}" not found. ` +
        `Available: ${tools.map((t) => t.function.name).join(', ')}`,
      );
    }
    return buildSingleToolSchema(tool);
  }

  // 'required' — any tool
  if (tools.length === 1) {
    return buildSingleToolSchema(tools[0]);
  }

  return { oneOf: tools.map(buildSingleToolSchema) };
}

/**
 * Parse model output as tool calls. Handles raw JSON, markdown code blocks,
 * and JSON embedded in text. Returns null if the output is regular text.
 */
function parseToolCalls(
  output: string,
  tools: ChatCompletionTool[],
): ChatCompletionToolCall[] | null {
  const trimmed = output.trim();
  if (!trimmed) return null;

  let parsed: unknown = tryParseJson(trimmed);

  if (parsed === undefined) {
    // Try markdown code block
    const codeBlock = trimmed.match(/```(?:json)?\s*\n?([\s\S]*?)\n?\s*```/);
    if (codeBlock) parsed = tryParseJson(codeBlock[1].trim());
  }

  if (parsed === undefined) {
    // Try extracting the outermost JSON object or array
    const jsonMatch = trimmed.match(/(\{[\s\S]*\}|\[[\s\S]*\])/);
    if (jsonMatch) parsed = tryParseJson(jsonMatch[1]);
  }

  if (parsed === undefined || typeof parsed !== 'object' || parsed === null) return null;

  const items: unknown[] = Array.isArray(parsed) ? parsed : [parsed];
  const toolNames = new Set(tools.map((t) => t.function.name));
  const calls: ChatCompletionToolCall[] = [];

  for (const item of items) {
    if (!isToolCallShape(item, toolNames)) continue;
    const obj = item as { name: string; arguments: unknown };
    calls.push({
      id: `call-${randomUUID()}`,
      type: 'function',
      function: {
        name: obj.name,
        arguments: typeof obj.arguments === 'string'
          ? obj.arguments
          : JSON.stringify(obj.arguments ?? {}),
      },
    });
  }

  return calls.length > 0 ? calls : null;
}

function tryParseJson(text: string): unknown | undefined {
  try {
    return JSON.parse(text);
  } catch {
    return undefined;
  }
}

function isToolCallShape(value: unknown, validNames: Set<string>): boolean {
  if (typeof value !== 'object' || value === null) return false;
  const obj = value as Record<string, unknown>;
  return typeof obj.name === 'string' && validNames.has(obj.name) && 'arguments' in obj;
}
