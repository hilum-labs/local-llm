// ── Tool definitions ─────────────────────────────────────────────────────────

export interface ChatCompletionTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    /** JSON Schema for the function's parameters. */
    parameters?: Record<string, unknown>;
  };
}

export type ChatCompletionToolChoice =
  | 'auto'
  | 'none'
  | 'required'
  | { type: 'function'; function: { name: string } };

export interface ChatCompletionToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    /** JSON-encoded arguments string. */
    arguments: string;
  };
}

// ── Request ──────────────────────────────────────────────────────────────────

export interface ChatCompletionResponseFormat {
  type: 'text' | 'json_object' | 'json_schema';
  json_schema?: {
    name?: string;
    strict?: boolean;
    schema: Record<string, unknown>;
  };
}

export interface ChatCompletionRequestMessage {
  role: string;
  content: string | null | Array<{
    type: 'text' | 'image_url';
    text?: string;
    image_url?: { url: string; detail?: string };
  }>;
  /** Tool calls made by the assistant (present on assistant messages). */
  tool_calls?: ChatCompletionToolCall[];
  /** ID of the tool call this message is responding to (present on tool messages). */
  tool_call_id?: string;
  /** Function name for tool result messages. */
  name?: string;
}

export interface ChatCompletionRequest {
  messages: ChatCompletionRequestMessage[];
  model?: string;
  temperature?: number;
  top_p?: number;
  max_tokens?: number;
  stream?: boolean;
  stop?: string[];
  frequency_penalty?: number;
  presence_penalty?: number;
  top_k?: number;
  seed?: number;
  /** Constrain output format — 'json_object' for free JSON, 'json_schema' for schema-constrained. */
  response_format?: ChatCompletionResponseFormat;
  /** Raw GBNF grammar string for advanced output constraints. */
  grammar?: string;
  /** Per-request context overflow strategy — overrides the instance-level setting. */
  context_overflow?: 'error' | 'truncate_oldest' | 'sliding_window';
  /** Tool definitions available for the model to call. */
  tools?: ChatCompletionTool[];
  /** Controls which (if any) tool is called. Default: 'auto' when tools are present. */
  tool_choice?: ChatCompletionToolChoice;
}

// ── Response ─────────────────────────────────────────────────────────────────

export interface ChatCompletionMessage {
  role: 'assistant';
  content: string | null;
  tool_calls?: ChatCompletionToolCall[];
}

export interface ChatCompletionChoice {
  index: number;
  message: ChatCompletionMessage;
  finish_reason: 'stop' | 'length' | 'tool_calls';
}

export interface ChatCompletionUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface ChatCompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: ChatCompletionChoice[];
  usage: ChatCompletionUsage;
  /** Context window management metadata (present when overflow management is active). */
  _context?: import('./types.js').ContextMetadata;
  /** Inference performance metrics (prompt eval time, generation speed, etc.). */
  _timing?: import('./types.js').InferenceMetrics;
}

// ── Streaming ────────────────────────────────────────────────────────────────

export interface ChatCompletionToolCallDelta {
  index: number;
  id?: string;
  type?: 'function';
  function?: {
    name?: string;
    arguments?: string;
  };
}

export interface ChatCompletionChunkDelta {
  role?: 'assistant';
  content?: string;
  tool_calls?: ChatCompletionToolCallDelta[];
}

export interface ChatCompletionChunkChoice {
  index: number;
  delta: ChatCompletionChunkDelta;
  finish_reason: 'stop' | 'length' | 'tool_calls' | null;
}

export interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: ChatCompletionChunkChoice[];
  /** Context window management metadata (present on the first chunk when overflow management is active). */
  _context?: import('./types.js').ContextMetadata;
  /** Inference performance metrics (present on the final chunk with finish_reason). */
  _timing?: import('./types.js').InferenceMetrics;
}
