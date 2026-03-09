# API Reference

## LocalLLM

The main class that provides an OpenAI-compatible interface to local LLM inference.

### `LocalLLM.create(options)`

Static async factory method. Creates and initializes a `LocalLLM` instance.

```typescript
const ai = await LocalLLM.create(options);
```

**Parameters:**

| Option | Type | Default | Description |
|---|---|---|---|
| `model` | `string` | *required* | HuggingFace URL, shorthand (`user/repo/file.gguf`), or local file path |
| `projector` | `string` | — | Path to mmproj GGUF file (required for vision models). URL, shorthand, or local path. |
| `compute` | `ComputeMode` | `'auto'` | `'auto'`, `'gpu'`, `'cpu'`, or `'hybrid'` |
| `gpuLayers` | `number` | — | Override GPU layer count directly (advanced) |
| `useMmap` | `boolean` | `true` | Memory-map the model file |
| `contextSize` | `number` | model default | Context window size in tokens |
| `batchSize` | `number` | — | Batch size for prompt processing |
| `threads` | `number` | — | CPU thread count |
| `cacheDir` | `string` | `~/.local-llm/models` | Directory for cached model downloads |
| `onProgress` | `(percent: number) => void` | — | Download progress callback (0-100) |
| `contextOverflow` | `ContextOverflowStrategy \| ContextOverflowConfig` | `'sliding_window'` | Context overflow strategy or full config object. Active by default — context size is auto-detected from the native context. |
| `warmup` | `boolean` | `true` | Run a single-token warmup pass after model load to prime GPU shaders and CPU caches. Eliminates 500ms–2s cold-start on first inference. Set `false` to defer context creation to first use. |
| `draftModel` | `string` | — | Path to a small draft model for speculative decoding. Must share the same tokenizer as the main model (e.g. same model family). When set, generation uses the draft to predict multiple tokens at once, then verifies in batch — typically 2-3x faster. |
| `draftNMax` | `number` | `16` | Max draft tokens per speculative step. Higher values give more acceleration when the draft model is accurate. |

**Returns:** `Promise<LocalLLM>`

### `new LocalLLM(options)`

Constructor. Creates an uninitialized instance. You must call `init()` before using it.

```typescript
const ai = new LocalLLM(options);
await ai.init();
```

### `ai.init()`

Initializes the instance: downloads the model (if URL), loads it into memory, and creates a context. Safe to call multiple times — subsequent calls are no-ops.

**Returns:** `Promise<void>`

### `LocalLLM.preload(model, options?)`

Static method. Pre-downloads a model to the local cache **without** loading it into memory. Call this early (e.g. at app startup) so that `create()` later skips the download and only does the fast load-from-disk step.

```typescript
// Start downloading at app startup — don't await, don't block
LocalLLM.preload('user/repo/model.gguf', {
  onProgress: (pct) => console.log(`${pct.toFixed(0)}%`),
})

// Later, when AI is needed — model is already cached, create() is fast
const ai = await LocalLLM.create({ model: 'user/repo/model.gguf' })
```

| Option | Type | Default | Description |
|---|---|---|---|
| `cacheDir` | `string` | `~/.local-llm/models` | Cache directory |
| `onProgress` | `(percent: number) => void` | — | Download progress callback (0-100) |

**Returns:** `Promise<string>` — local file path

Local file paths (`./model.gguf`, `/path/to/model.gguf`) are returned unchanged without downloading. Cached models resolve immediately.

### `ai.chat.completions`

OpenAI-compatible chat completions interface. See [ChatCompletions](#chatcompletions).

### `ai.dispose()`

Frees all native resources (model and context). Safe to call multiple times.

Also implements `Symbol.dispose` for use with `using`:

```typescript
using ai = await LocalLLM.create({ model: '...' });
// automatically disposed when leaving scope
```

---

## ChatCompletions

Accessed via `ai.chat.completions`. Provides the OpenAI-compatible `create()` method.

### `ai.chat.completions.create(params)`

Creates a chat completion.

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `messages` | `ChatMessage[]` | *required* | Array of `{ role, content }` messages. Content can be a string or an array of content parts (text + images). |
| `model` | `string` | — | Ignored (model is set in constructor) |
| `max_tokens` | `number` | `512` | Maximum tokens to generate |
| `temperature` | `number` | `0.7` | Sampling temperature (0.0 - 2.0) |
| `top_p` | `number` | `0.9` | Nucleus sampling threshold |
| `top_k` | `number` | `40` | Top-k sampling |
| `frequency_penalty` | `number` | `0` | Penalizes tokens proportional to frequency (OpenAI-compatible, 0–2) |
| `presence_penalty` | `number` | `0` | Penalizes tokens that have appeared at all (OpenAI-compatible, 0–2) |
| `seed` | `number` | — | Random seed for reproducibility |
| `stream` | `boolean` | `false` | Enable streaming |
| `response_format` | `ChatCompletionResponseFormat` | — | Constrain output format (see below) |
| `grammar` | `string` | — | Raw GBNF grammar string for advanced constraints |
| `context_overflow` | `ContextOverflowStrategy` | — | Per-request override for context overflow strategy |
| `tools` | `ChatCompletionTool[]` | — | Tool definitions available for the model to call |
| `tool_choice` | `ChatCompletionToolChoice` | `'auto'` | Controls which tool is called: `'auto'`, `'none'`, `'required'`, or a specific function |

#### Function / Tool Calling

Enable the model to call structured functions. Built on top of grammar-constrained output — when `tool_choice` is `'required'` or a specific function, the model is grammar-constrained to produce valid tool call JSON.

```typescript
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'What is the weather in Paris?' }],
  tools: [
    {
      type: 'function',
      function: {
        name: 'get_weather',
        description: 'Get the current weather for a location',
        parameters: {
          type: 'object',
          properties: {
            location: { type: 'string', description: 'City name' },
            unit: { type: 'string', enum: ['celsius', 'fahrenheit'] },
          },
          required: ['location'],
        },
      },
    },
  ],
});

if (response.choices[0].finish_reason === 'tool_calls') {
  const toolCalls = response.choices[0].message.tool_calls!;
  console.log(toolCalls[0].function.name);       // "get_weather"
  console.log(toolCalls[0].function.arguments);   // '{"location":"Paris"}'
}
```

**Tool choice modes:**

| `tool_choice` | Behavior |
|---|---|
| `'auto'` (default) | Model decides whether to call a tool or respond with text. No grammar constraint. |
| `'none'` | Tools are ignored, model responds normally |
| `'required'` | Model must call one of the provided tools (grammar-constrained) |
| `{ type: 'function', function: { name: '...' } }` | Model must call the specified function (grammar-constrained) |

**Multi-turn tool use:**

After receiving a tool call, execute the function and send the result back:

```typescript
const messages = [
  { role: 'user', content: 'What is the weather in Paris and Tokyo?' },
];

// First turn — model calls a tool
const res1 = await ai.chat.completions.create({ messages, tools });
const call = res1.choices[0].message.tool_calls![0];

// Execute the tool
const weatherResult = await getWeather(JSON.parse(call.function.arguments));

// Second turn — send the result back
messages.push(
  { role: 'assistant', content: null, tool_calls: res1.choices[0].message.tool_calls },
  { role: 'tool', content: JSON.stringify(weatherResult), tool_call_id: call.id, name: call.function.name },
);

const res2 = await ai.chat.completions.create({ messages, tools });
console.log(res2.choices[0].message.content); // Natural language response using the tool result
```

**Streaming with tools:**

When tools are present, the stream buffers internally to determine if the output is a tool call or regular text, then emits the appropriate OpenAI-compatible delta format:

```typescript
const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'What is the weather in Paris?' }],
  tools: [...],
  tool_choice: 'required',
  stream: true,
});

for await (const chunk of stream) {
  if (chunk.choices[0].delta.tool_calls) {
    console.log(chunk.choices[0].delta.tool_calls);
  }
  if (chunk.choices[0].finish_reason === 'tool_calls') {
    console.log('Tool call complete');
  }
}
```

#### Context Window Management

Context overflow management is **active by default** — the actual context window size is auto-detected from the native llama.cpp context via `llama_n_ctx()`, so you don't need to specify `contextSize` manually.

**Default behavior:** `sliding_window` strategy with a 25% reserve ratio for generation tokens.

```typescript
// Simple — defaults are already active
const ai = await LocalLLM.create({
  model: 'user/repo/model.gguf',
});

// Explicit strategy
const ai = await LocalLLM.create({
  model: 'user/repo/model.gguf',
  contextOverflow: 'truncate_oldest',
});

// Full config with reserve ratio and overflow callback
const ai = await LocalLLM.create({
  model: 'user/repo/model.gguf',
  contextOverflow: {
    strategy: 'sliding_window',
    reserveRatio: 0.3, // reserve 30% for generation
    onOverflow: async (event) => {
      console.log(`Overflow: ${event.promptTokens} tokens > ${event.availableTokens} budget`);
      // Optionally return replacement messages (e.g. summarized conversation)
      // return [{ role: 'system', content: summary }, ...recentMessages];
    },
  },
});
```

| Strategy | Behavior |
|---|---|
| `'error'` | Throws an error when messages exceed the context limit |
| `'truncate_oldest'` | Drops the oldest non-system messages until the conversation fits |
| `'sliding_window'` | Keeps system message(s) + the most recent messages that fit (default) |

**How it works:**

1. **Auto-detect context size** — reads `llama_n_ctx()` from the native context (no manual `contextSize` needed)
2. **Smart token budgeting** — tokenizes the full formatted prompt (via `applyChatTemplate` + `tokenize`) instead of estimating per-message, accounting for template tokens and special tokens
3. **Reserve ratio** — when `max_tokens` is not set, reserves `contextSize * reserveRatio` tokens for generation (default 25%)
4. **Overflow callback** — fires `onOverflow` before applying the strategy, giving you a chance to return summarized messages
5. **Binary search trimming** — uses binary search over message windows for efficient truncation
6. **Transparency metadata** — every response includes a `_context` field with full context management details

System messages are always preserved. You can override the strategy per-request:

```typescript
const response = await ai.chat.completions.create({
  messages: longConversation,
  context_overflow: 'truncate_oldest',
});

// Access context metadata
console.log(response._context);
// {
//   overflowTriggered: true,
//   strategyUsed: 'truncate_oldest',
//   originalMessageCount: 50,
//   keptMessageCount: 12,
//   contextSize: 4096,
//   promptTokens: 3072,
//   reservedForGeneration: 1024,
// }
```

#### Structured Output / Response Format

Constrain model output to valid JSON or a specific JSON Schema:

```typescript
// Free-form JSON output
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'List 3 colors as JSON' }],
  response_format: { type: 'json_object' },
});

// Schema-constrained JSON output
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Describe a person' }],
  response_format: {
    type: 'json_schema',
    json_schema: {
      name: 'person',
      schema: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          age: { type: 'integer' },
        },
        required: ['name', 'age'],
      },
    },
  },
});
```

You can also pass a raw GBNF grammar string for full control:

```typescript
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Say yes or no' }],
  grammar: 'root ::= "yes" | "no"',
});
```

#### Non-streaming (default)

```typescript
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello' }],
  max_tokens: 128,
});
```

**Returns:** `Promise<ChatCompletionResponse>`

```typescript
interface ChatCompletionResponse {
  id: string;                    // "chatcmpl-<uuid>"
  object: 'chat.completion';
  created: number;               // Unix timestamp
  model: string;                 // Model filename
  choices: [{
    index: number;
    message: {
      role: 'assistant';
      content: string;
    };
    finish_reason: 'stop' | 'length';
  }];
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  _context?: ContextMetadata;    // Context window management details
  _timing?: InferenceMetrics;    // Performance metrics (prompt eval, generation speed)
}
```

#### Streaming (`stream: true`)

```typescript
const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello' }],
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
}
```

**Returns:** `Promise<AsyncIterable<ChatCompletionChunk>>`

```typescript
interface ChatCompletionChunk {
  id: string;
  object: 'chat.completion.chunk';
  created: number;
  model: string;
  choices: [{
    index: number;
    delta: {
      role?: 'assistant';     // First chunk only
      content?: string;       // Content chunks
    };
    finish_reason: 'stop' | 'length' | null;  // Last chunk only
  }];
  _timing?: InferenceMetrics;  // Present on the final chunk
}
```

Chunk sequence:
1. `{ delta: { role: 'assistant' }, finish_reason: null }` — first chunk
2. `{ delta: { content: '...' }, finish_reason: null }` — content chunks (one per token)
3. `{ delta: {}, finish_reason: 'stop', _timing: {...} }` — final chunk (includes performance metrics)

---

## Embeddings

Accessed via `ai.embeddings`. Provides the OpenAI-compatible embeddings endpoint. Requires `embeddings: true` in options.

### Setup

```typescript
const ai = await LocalLLM.create({
  model: './model.gguf',
  embeddings: true,
});

// Or with custom pooling:
const ai = await LocalLLM.create({
  model: './model.gguf',
  embeddings: { poolingType: 'cls' },
});
```

| Option | Type | Default | Description |
|---|---|---|---|
| `embeddings` | `boolean \| { poolingType? }` | `false` | Enable embeddings API |
| `poolingType` | `EmbeddingPoolingType` | `'mean'` | `'mean'`, `'cls'`, or `'last'` |

### `ai.embeddings.create(params)`

Creates embeddings for one or more text inputs.

**Parameters:**

| Param | Type | Default | Description |
|---|---|---|---|
| `input` | `string \| string[]` | *required* | Text or texts to embed |
| `model` | `string` | — | Ignored (model is set in constructor) |
| `encoding_format` | `'float' \| 'base64'` | `'float'` | Output format for embedding vectors |

**Returns:** `Promise<EmbeddingResponse>`

```typescript
// Single text
const result = await ai.embeddings.create({
  input: 'What is TypeScript?',
});

console.log(result.data[0].embedding.length); // e.g. 2048
console.log(result.usage.prompt_tokens);

// Batch (more efficient — single forward pass)
const batch = await ai.embeddings.create({
  input: ['First text', 'Second text', 'Third text'],
});
```

**Response format:**

```typescript
interface EmbeddingResponse {
  object: 'list';
  model: string;
  data: {
    object: 'embedding';
    index: number;
    embedding: number[] | string; // number[] for float, base64 string for base64
  }[];
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}
```

Embeddings are **L2-normalized** by default, so dot product equals cosine similarity.

---

## Model

Lower-level class for direct model access. Use this when you need tokenization, chat templates, or multiple contexts.

### `new Model(path, options?)`

```typescript
const model = new Model('/path/to/model.gguf', {
  compute: 'gpu',
  gpuLayers: 99,
  useMmap: true,
});
```

| Option | Type | Default | Description |
|---|---|---|---|
| `compute` | `ComputeMode` | `'auto'` | `'auto'`, `'gpu'`, `'cpu'`, or `'hybrid'` |
| `gpuLayers` | `number` | — | Override GPU layer count |
| `useMmap` | `boolean` | `true` | Memory-map the model file |

### `model.loadProjector(projectorPath, options?)`

Loads a vision projector (mmproj GGUF file) for multimodal support.

```typescript
model.loadProjector('./mmproj-model.gguf');
```

| Option | Type | Default | Description |
|---|---|---|---|
| `useGpu` | `boolean` | — | Use GPU for vision encoding |
| `threads` | `number` | — | Thread count for vision encoding |

### `model.isMultimodal`

`boolean` — `true` if a projector has been loaded.

### `model.createContext(options?)`

Creates an inference context.

```typescript
const ctx = model.createContext({
  contextSize: 4096,
  batchSize: 512,
  threads: 4,
});
```

**Returns:** `InferenceContext`

### `model.tokenize(text)`

Converts text to token IDs.

```typescript
const tokens: Int32Array = model.tokenize('Hello world');
```

### `model.detokenize(tokens)`

Converts token IDs back to text.

```typescript
const text: string = model.detokenize(tokens);
```

### `model.applyChatTemplate(messages, addAssistant?)`

Applies the model's built-in chat template to format messages into a prompt string.

```typescript
const prompt = model.applyChatTemplate([
  { role: 'system', content: 'You are helpful.' },
  { role: 'user', content: 'Hello' },
], true); // true = add assistant turn prefix
```

| Param | Type | Default | Description |
|---|---|---|---|
| `messages` | `ChatMessage[]` | *required* | Messages to format |
| `addAssistant` | `boolean` | `false` | Append assistant turn prefix |

**Returns:** `string`

### `model.embeddingDimension`

`number` — The dimensionality of embedding vectors produced by this model.

### `model.createEmbeddingContext(options?)`

Creates a context for computing embeddings.

```typescript
const embCtx = model.createEmbeddingContext({
  poolingType: 'mean',
  contextSize: 4096,
});
```

| Option | Type | Default | Description |
|---|---|---|---|
| `poolingType` | `EmbeddingPoolingType` | `'mean'` | `'mean'`, `'cls'`, or `'last'` |
| `contextSize` | `number` | model default | Context window for embedding |
| `batchSize` | `number` | — | Batch size |
| `threads` | `number` | — | CPU thread count |

**Returns:** `EmbeddingContext`

### `model.dispose()`

Frees the model from memory. Safe to call multiple times.

---

## EmbeddingContext

Embedding context created by `Model.createEmbeddingContext()`. Do not construct directly.

### `embCtx.dimension`

`number` — The dimensionality of embedding vectors.

### `embCtx.embed(text)`

Embeds a single text string. Returns an L2-normalized `Float32Array`.

```typescript
const vec = embCtx.embed('Hello, world!');
console.log(vec.length); // e.g. 2048
```

**Returns:** `Float32Array`

### `embCtx.embedBatch(texts)`

Embeds multiple texts in a single forward pass. More efficient than calling `embed()` in a loop.

```typescript
const vecs = embCtx.embedBatch(['First text', 'Second text', 'Third text']);
console.log(vecs.length); // 3
```

**Returns:** `Float32Array[]`

### `embCtx.dispose()`

Frees the embedding context. Safe to call multiple times.

---

## InferenceContext

Inference context created by `Model.createContext()`. Do not construct directly.

### `ctx.contextSize`

`number` — The actual context window size in tokens, as allocated by llama.cpp. Reads `llama_n_ctx()` from the native context.

```typescript
const ctx = model.createContext({ contextSize: 4096 });
console.log(ctx.contextSize); // 4096 (or the model's default if not specified)
```

### `ctx.generate(prompt, options?)`

Generates a complete response.

```typescript
const result = await ctx.generate(prompt, {
  maxTokens: 256,
  temperature: 0.7,
  topP: 0.9,
  topK: 40,
  repeatPenalty: 1.1,
  seed: 42,
});
```

**Returns:** `Promise<string>`

### `ctx.stream(prompt, options?)`

Streams tokens as an async generator.

```typescript
for await (const token of ctx.stream(prompt, { maxTokens: 256 })) {
  process.stdout.write(token);
}
```

**Returns:** `AsyncGenerator<string>`

### `ctx.generateVision(prompt, imageBuffers, options?)`

Generates a complete response from a prompt with images. The prompt must contain `<__media__>` markers for each image.

```typescript
const result = await ctx.generateVision(prompt, [imageBuffer], { maxTokens: 256 });
```

**Returns:** `Promise<string>`

### `ctx.streamVision(prompt, imageBuffers, options?)`

Streams tokens from a vision prompt with images.

```typescript
for await (const token of ctx.streamVision(prompt, [imageBuffer], { maxTokens: 256 })) {
  process.stdout.write(token);
}
```

**Returns:** `AsyncGenerator<string>`

### `ctx.getPerf()`

Returns performance metrics from the most recent inference call. Returns `null` if the native backend doesn't support it.

```typescript
const text = await ctx.generate(prompt, { maxTokens: 128 });
const perf = ctx.getPerf();

console.log(`Prompt eval: ${perf.promptTokensPerSec.toFixed(0)} tok/s (${perf.promptEvalMs.toFixed(0)} ms)`);
console.log(`Generation:  ${perf.generatedTokensPerSec.toFixed(0)} tok/s (${perf.generationMs.toFixed(0)} ms)`);
console.log(`Tokens: ${perf.promptTokens} prompt, ${perf.generatedTokens} generated`);
```

**Returns:** `InferenceMetrics | null`

### `ctx.benchmark(options?)`

Runs a reproducible benchmark: evaluates a synthetic prompt then generates tokens, repeating for the requested number of iterations. Clears KV cache between runs.

```typescript
const result = await ctx.benchmark({
  promptTokens: 256,
  generateTokens: 128,
  iterations: 5,
});

console.log(`Prompt eval: ${result.promptTokensPerSec.toFixed(0)} tok/s`);
console.log(`Generation:  ${result.generatedTokensPerSec.toFixed(0)} tok/s`);
console.log(`TTFT:        ${result.ttftMs.toFixed(0)} ms`);
console.log(`Total:       ${result.totalMs.toFixed(0)} ms (${result.iterations} iterations)`);
```

**Parameters:**

| Option | Type | Default | Description |
|---|---|---|---|
| `promptTokens` | `number` | `128` | Number of prompt tokens to evaluate |
| `generateTokens` | `number` | `64` | Number of tokens to generate per iteration |
| `iterations` | `number` | `3` | Number of benchmark iterations |

**Returns:** `Promise<BenchmarkResult>`

### `ctx.dispose()`

Frees the context. Safe to call multiple times.

---

## ModelManager

Downloads and caches GGUF models from HuggingFace.

### `new ModelManager(cacheDir?)`

```typescript
const manager = new ModelManager();
// or
const manager = new ModelManager('/custom/cache/dir');
```

Default cache directory: `~/.local-llm/models/`

### `manager.downloadModel(url, options?)`

Downloads a model or returns the cached path if already downloaded.

```typescript
const path = await manager.downloadModel(
  'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  {
    onProgress: (downloaded, total, percent) => {
      console.log(`${percent.toFixed(1)}%`);
    },
  },
);
```

**URL formats:**
- Full URL: `https://huggingface.co/user/repo/resolve/main/file.gguf`
- Shorthand: `user/repo/file.gguf` (resolves to `main` branch)
- Shorthand with branch: `user/repo/branch/file.gguf`

**Returns:** `Promise<string>` — local file path

### `manager.listModels()`

Lists all cached models.

```typescript
const models: CacheEntry[] = await manager.listModels();
```

```typescript
interface CacheEntry {
  url: string;
  path: string;
  size: number;
  downloadedAt: string;
  lastUsedAt: string;
}
```

### `manager.removeModel(url)`

Removes a cached model by its original URL. Deletes both the file and the cache index entry.

```typescript
const removed: boolean = await manager.removeModel('https://huggingface.co/...');
```

---

## Types

### `ComputeMode`

```typescript
type ComputeMode = 'auto' | 'cpu' | 'hybrid' | 'gpu';
```

| Mode | Behavior |
|---|---|
| `'auto'` | Detects available GPU backend (Metal/CUDA/Vulkan). Falls back to CPU. |
| `'gpu'` | Offload all layers to GPU |
| `'hybrid'` | Partial GPU offload (16 layers) |
| `'cpu'` | CPU only, no GPU |

### `ChatMessage`

```typescript
interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string | ContentPart[];  // string for text-only, array for multimodal
}
```

### `ContentPart`

```typescript
type ContentPart = TextContentPart | ImageContentPart;

interface TextContentPart {
  type: 'text';
  text: string;
}

interface ImageContentPart {
  type: 'image_url';
  image_url: { url: string };  // data: URI, file path, or HTTP URL
}
```

### `GenerateOptions`

```typescript
interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;       // Multiplicative penalty for repeated tokens (llama.cpp-specific, 1.0–1.3)
  frequencyPenalty?: number;    // Additive penalty proportional to frequency (OpenAI-compatible, 0–2)
  presencePenalty?: number;     // Additive penalty for any appeared token (OpenAI-compatible, 0–2)
  seed?: number;
  stop?: string[];
  signal?: AbortSignal;
  grammar?: string;
  grammarRoot?: string;
  responseFormat?: ResponseFormat;
  onPromptProgress?: (processed: number, total: number) => boolean;
}
```

| Option | Type | Default | Description |
|---|---|---|---|
| `onPromptProgress` | `(processed, total) => boolean` | — | Called between prompt evaluation chunks with the number of tokens processed and total. Return `false` to abort generation during prompt eval. Useful for progress bars or cancelling long prompts. |

### `ResponseFormat`

```typescript
interface ResponseFormat {
  type: 'text' | 'json_object' | 'json_schema';
  json_schema?: {
    name?: string;
    strict?: boolean;
    schema: Record<string, unknown>;
  };
}
```

### `EmbeddingPoolingType`

```typescript
type EmbeddingPoolingType = 'mean' | 'cls' | 'last';
```

| Pooling | How it works | Best for |
|---|---|---|
| `'mean'` | Average all token vectors | General purpose (default) |
| `'cls'` | Use first token's vector | BERT-style models |
| `'last'` | Use last token's vector | Some newer models (e.g. Mistral embeddings) |

### `ContextOverflowStrategy`

```typescript
type ContextOverflowStrategy = 'error' | 'truncate_oldest' | 'sliding_window';
```

| Strategy | Behavior |
|---|---|
| `'error'` | Throws an error when messages exceed the context limit |
| `'truncate_oldest'` | Drops oldest non-system messages until conversation fits |
| `'sliding_window'` | Keeps system messages + most recent messages that fit (default) |

### `ContextOverflowConfig`

Full configuration object for context overflow management:

```typescript
interface ContextOverflowConfig {
  strategy?: ContextOverflowStrategy;          // default: 'sliding_window'
  reserveRatio?: number;                       // default: 0.25 (25%)
  onOverflow?: (event: ContextOverflowEvent) =>
    OverflowMessage[] | void | Promise<OverflowMessage[] | void>;
}
```

| Field | Type | Default | Description |
|---|---|---|---|
| `strategy` | `ContextOverflowStrategy` | `'sliding_window'` | Which trimming strategy to apply |
| `reserveRatio` | `number` | `0.25` | Fraction of context reserved for generation when `max_tokens` is not set |
| `onOverflow` | `function` | — | Callback fired before the strategy is applied. Return replacement messages to override. |

### `ContextOverflowEvent`

Passed to the `onOverflow` callback:

```typescript
interface ContextOverflowEvent {
  messages: OverflowMessage[];
  originalMessageCount: number;
  promptTokens: number;
  availableTokens: number;
  contextSize: number;
  reservedForGeneration: number;
  strategy: ContextOverflowStrategy;
}
```

### `ContextMetadata`

Attached to every response as `_context`:

```typescript
interface ContextMetadata {
  overflowTriggered: boolean;
  strategyUsed?: ContextOverflowStrategy;
  originalMessageCount: number;
  keptMessageCount: number;
  contextSize: number;
  promptTokens: number;
  reservedForGeneration: number;
}
```

### `InferenceMetrics`

Performance metrics from a single inference call. Attached to responses as `_timing`:

```typescript
interface InferenceMetrics {
  promptEvalMs: number;        // time to process prompt tokens (ms)
  generationMs: number;        // time to generate completion tokens (ms)
  promptTokens: number;        // number of prompt tokens processed
  generatedTokens: number;     // number of tokens generated
  promptTokensPerSec: number;  // prompt processing speed
  generatedTokensPerSec: number; // token generation speed
}
```

### `BenchmarkOptions`

Options for `ctx.benchmark()`:

```typescript
interface BenchmarkOptions {
  promptTokens?: number;   // default: 128
  generateTokens?: number; // default: 64
  iterations?: number;     // default: 3
}
```

### `BenchmarkResult`

Result from `ctx.benchmark()`:

```typescript
interface BenchmarkResult {
  promptTokensPerSec: number;    // averaged across iterations
  generatedTokensPerSec: number; // averaged across iterations
  ttftMs: number;                // time-to-first-token averaged
  totalMs: number;               // total wall time
  iterations: number;
  individual: InferenceMetrics[]; // per-iteration data
}
```

### `OverflowMessage`

Simplified message type used by the overflow callback:

```typescript
interface OverflowMessage {
  role: string;
  content: string | Array<{ type: string; text?: string; image_url?: { url: string; detail?: string } }>;
}
```

### `ChatCompletionTool`

```typescript
interface ChatCompletionTool {
  type: 'function';
  function: {
    name: string;
    description?: string;
    parameters?: Record<string, unknown>;  // JSON Schema
  };
}
```

### `ChatCompletionToolChoice`

```typescript
type ChatCompletionToolChoice =
  | 'auto'       // model decides (default)
  | 'none'       // don't call any tool
  | 'required'   // must call a tool (grammar-constrained)
  | { type: 'function'; function: { name: string } };  // must call this specific function
```

### `ChatCompletionToolCall`

Present in assistant messages when the model calls a tool:

```typescript
interface ChatCompletionToolCall {
  id: string;              // "call-<uuid>"
  type: 'function';
  function: {
    name: string;
    arguments: string;     // JSON-encoded arguments
  };
}
```

---

## LocalLLMProvider

Vercel AI SDK provider backed by a local llama.cpp model. Implements the `LanguageModelV3` interface.

### Getting an instance

```typescript
const ai = await LocalLLM.create({ model: '...' });
const lm = ai.languageModel();       // default modelId
const lm = ai.languageModel('my-id'); // custom modelId
```

### Properties

| Property | Type | Description |
|---|---|---|
| `specificationVersion` | `'v3'` | AI SDK specification version |
| `provider` | `'local-llm'` | Provider identifier |
| `modelId` | `string` | Model identifier (filename or custom) |
| `supportedUrls` | `Record<string, RegExp[]>` | Always `{}` (no URL-based content) |

### `doGenerate(options)`

Non-streaming generation. Called internally by `generateText()` — you typically don't call this directly.

```typescript
import { generateText } from 'ai';

const { text, usage } = await generateText({
  model: ai.languageModel(),
  prompt: 'Hello',
  maxTokens: 128,
});
```

**Returns:** `Promise<{ content, finishReason, usage, warnings }>`

### `doStream(options)`

Streaming generation. Called internally by `streamText()` — you typically don't call this directly.

```typescript
import { streamText } from 'ai';

const result = streamText({
  model: ai.languageModel(),
  prompt: 'Hello',
  maxTokens: 128,
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
```

**Returns:** `Promise<{ stream: ReadableStream }>`

Stream part sequence:
1. `{ type: 'stream-start', warnings }` — stream initialization
2. `{ type: 'text-start', id }` — text block begins
3. `{ type: 'text-delta', textDelta }` — content chunks (one per token)
4. `{ type: 'text-end' }` — text block ends
5. `{ type: 'finish', finishReason, usage }` — generation complete

### Supported and unsupported features

| Feature | Supported | Notes |
|---|---|---|
| `maxOutputTokens` | Yes | Maps to `maxTokens` |
| `temperature` | Yes | |
| `topP` | Yes | |
| `topK` | Yes | |
| `frequencyPenalty` | Yes | Maps to llama.cpp `penalty_freq` |
| `presencePenalty` | Yes | Maps to llama.cpp `penalty_present` |
| `seed` | Yes | |
| `stopSequences` | Yes | |
| `responseFormat` (JSON) | Yes | Uses GBNF grammar constraints |
| `responseFormat` (JSON Schema) | Yes | Schema compiled to GBNF grammar |
| `tools` / `toolChoice` | No | Warning emitted |
| `presencePenalty` | No | Warning emitted |

---

## ModelPool

Shared in-process model pool with LRU eviction. Manages multiple loaded models with reference counting and automatic memory management.

### `new ModelPool(options?)`

```typescript
import { ModelPool } from 'local-llm';

const pool = new ModelPool({
  maxMemoryBytes: 8 * 1024 * 1024 * 1024, // 8 GB
  maxModels: 3,
});
```

| Option | Type | Default | Description |
|---|---|---|---|
| `maxMemoryBytes` | `number` | `0` (unlimited) | Maximum total memory for all loaded models |
| `maxModels` | `number` | `0` (unlimited) | Maximum number of models to keep loaded |

When limits are exceeded, the least-recently-used model with `refCount === 0` is evicted.

### `pool.load(alias, pathOrBuffer, options?)`

Loads a model into the pool or returns an existing one if already loaded (incrementing its reference count).

```typescript
const model = await pool.load('chat', './chat-model.gguf', { compute: 'gpu' });
```

**Returns:** `Promise<Model>`

### `pool.get(alias)`

Retrieves a loaded model by alias without incrementing the reference count. Updates `lastAccess`.

```typescript
const model = pool.get('chat'); // Model | undefined
```

### `pool.release(alias)`

Decrements the reference count. When it reaches 0, the model is disposed and removed from the pool.

```typescript
pool.release('chat'); // returns true if alias existed
```

### `pool.unload(alias)`

Force-unloads a model regardless of reference count.

```typescript
pool.unload('chat');
```

### `pool.list()`

Returns information about all loaded models.

```typescript
const info: PoolInfo[] = pool.list();
```

```typescript
interface PoolInfo {
  alias: string;
  sizeBytes: number;
  refCount: number;
  lastAccess: number;  // timestamp
}
```

### `pool.dispose()`

Disposes all loaded models and clears the pool. Safe to call multiple times.

### `LocalLLM.pool`

Static accessor for a shared `ModelPool` instance:

```typescript
const chat = await LocalLLM.pool.load('chat', './chat.gguf');
const code = await LocalLLM.pool.load('code', './code.gguf');
// ...
LocalLLM.pool.dispose();
```
