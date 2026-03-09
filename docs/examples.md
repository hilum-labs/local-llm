# Examples

## Basic Chat

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const response = await ai.chat.completions.create({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'What is TypeScript?' },
  ],
  max_tokens: 256,
});

console.log(response.choices[0].message.content);
ai.dispose();
```

## Vision / Multimodal

Send images alongside text. Requires a vision model and projector:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf',
  projector: 'Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf',
});

// Using a base64 data URI
const response = await ai.chat.completions.create({
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: 'What is in this image?' },
      { type: 'image_url', image_url: { url: 'data:image/png;base64,...' } },
    ],
  }],
  max_tokens: 256,
});

console.log(response.choices[0].message.content);

// Using a local file path
const response2 = await ai.chat.completions.create({
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: 'Describe this photo.' },
      { type: 'image_url', image_url: { url: './photo.jpg' } },
    ],
  }],
  max_tokens: 256,
});

console.log(response2.choices[0].message.content);

// Streaming with vision
const stream = await ai.chat.completions.create({
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: 'What colors do you see?' },
      { type: 'image_url', image_url: { url: './photo.jpg' } },
    ],
  }],
  max_tokens: 128,
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
}
console.log();

ai.dispose();
```

## Streaming Responses

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Write a short poem about programming.' }],
  max_tokens: 256,
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
}
console.log();

ai.dispose();
```

## Download with Progress

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  onProgress: (pct) => {
    process.stdout.write(`\rDownloading: ${pct.toFixed(1)}%`);
  },
});

console.log('\nModel ready!');
ai.dispose();
```

## Preloading a Model

Pre-download a model at app startup so it's cached when the user needs it. The app doesn't block — the download runs in the background.

```typescript
import { LocalLLM } from 'local-llm';

const MODEL = 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';

// Fire-and-forget at startup — download runs in the background
LocalLLM.preload(MODEL, {
  onProgress: (pct) => console.log(`Preloading: ${pct.toFixed(0)}%`),
});

// Later, when the user needs AI — model is already cached, this is fast
async function handleUserRequest() {
  const ai = await LocalLLM.create({ model: MODEL });
  const response = await ai.chat.completions.create({
    messages: [{ role: 'user', content: 'Hello!' }],
    max_tokens: 128,
  });
  console.log(response.choices[0].message.content);
  ai.dispose();
}
```

## Using a Local Model File

```typescript
import { LocalLLM } from 'local-llm';

// Skip download — use a model you already have
const ai = await LocalLLM.create({
  model: './models/my-model.gguf',
  compute: 'gpu',
});

const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello!' }],
});

console.log(response.choices[0].message.content);
ai.dispose();
```

## Compute Modes

```typescript
import { LocalLLM } from 'local-llm';

// Auto-detect GPU (default)
const ai = await LocalLLM.create({
  model: './model.gguf',
  compute: 'auto', // detects Metal on macOS, CUDA on Linux/Windows
});

// Force CPU only (slower, but works everywhere)
const aiCpu = await LocalLLM.create({
  model: './model.gguf',
  compute: 'cpu',
});

// Force full GPU offload
const aiGpu = await LocalLLM.create({
  model: './model.gguf',
  compute: 'gpu',
});

// Hybrid: split between CPU and GPU (limited VRAM)
const aiHybrid = await LocalLLM.create({
  model: './model.gguf',
  compute: 'hybrid',
});
```

## Multi-turn Conversation

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const messages = [
  { role: 'system' as const, content: 'You are a math tutor.' },
];

// Turn 1
messages.push({ role: 'user' as const, content: 'What is 2 + 2?' });
const r1 = await ai.chat.completions.create({ messages, max_tokens: 64 });
const a1 = r1.choices[0].message.content;
messages.push({ role: 'assistant' as const, content: a1 });
console.log('Assistant:', a1);

// Turn 2
messages.push({ role: 'user' as const, content: 'Now multiply that by 3.' });
const r2 = await ai.chat.completions.create({ messages, max_tokens: 64 });
console.log('Assistant:', r2.choices[0].message.content);

ai.dispose();
```

## Function / Tool Calling

Enable the model to call structured functions. The model generates a tool call with parsed arguments instead of plain text.

```typescript
import { LocalLLM } from 'local-llm';
import type { ChatCompletionTool, ChatCompletionRequestMessage } from 'local-llm';

const tools: ChatCompletionTool[] = [
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
];

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

// Auto mode — model decides whether to call a tool
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'What is the weather in Paris?' }],
  tools,
});

if (response.choices[0].finish_reason === 'tool_calls') {
  const call = response.choices[0].message.tool_calls![0];
  console.log(`Tool: ${call.function.name}(${call.function.arguments})`);
} else {
  console.log(response.choices[0].message.content);
}

ai.dispose();
```

## Tool Calling — Required Mode (Grammar-Constrained)

Force the model to call a tool. Output is grammar-constrained to valid JSON matching the tool schema.

```typescript
import { LocalLLM } from 'local-llm';
import type { ChatCompletionTool } from 'local-llm';

const tools: ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'search',
      description: 'Search the web',
      parameters: {
        type: 'object',
        properties: { query: { type: 'string' } },
        required: ['query'],
      },
    },
  },
];

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

// Required — model must call one of the tools
const res = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Find information about TypeScript 6.0' }],
  tools,
  tool_choice: 'required',
  max_tokens: 256,
});

const call = res.choices[0].message.tool_calls![0];
console.log(call.function.name);      // "search"
console.log(call.function.arguments); // '{"query":"TypeScript 6.0"}'

// Force a specific function
const res2 = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Tell me about Paris' }],
  tools,
  tool_choice: { type: 'function', function: { name: 'search' } },
  max_tokens: 256,
});

console.log(res2.choices[0].message.tool_calls![0].function.arguments);

ai.dispose();
```

## Multi-Turn Tool Use

Execute a tool and send the result back to the model for a natural language response.

```typescript
import { LocalLLM } from 'local-llm';
import type { ChatCompletionTool, ChatCompletionRequestMessage } from 'local-llm';

const tools: ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get weather for a city',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string' },
        },
        required: ['location'],
      },
    },
  },
];

function getWeather(location: string) {
  return { temperature: 22, condition: 'sunny', location };
}

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const messages: ChatCompletionRequestMessage[] = [
  { role: 'user', content: 'What is the weather in London?' },
];

// Turn 1: model calls the tool
const res1 = await ai.chat.completions.create({ messages, tools, tool_choice: 'required', max_tokens: 256 });
const call = res1.choices[0].message.tool_calls![0];

// Execute the tool
const result = getWeather(JSON.parse(call.function.arguments).location);
console.log(`Tool result: ${JSON.stringify(result)}`);

// Turn 2: send tool result back
messages.push(
  { role: 'assistant', content: null, tool_calls: res1.choices[0].message.tool_calls },
  { role: 'tool', content: JSON.stringify(result), tool_call_id: call.id, name: call.function.name },
);

const res2 = await ai.chat.completions.create({ messages, tools, max_tokens: 256 });
console.log(res2.choices[0].message.content);
// "The weather in London is sunny with a temperature of 22°C."

ai.dispose();
```

## Streaming Tool Calls

Tool calls can be streamed. The stream buffers internally and emits OpenAI-compatible `tool_calls` delta chunks.

```typescript
import { LocalLLM } from 'local-llm';
import type { ChatCompletionTool } from 'local-llm';

const tools: ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'calculate',
      description: 'Evaluate a math expression',
      parameters: {
        type: 'object',
        properties: { expression: { type: 'string' } },
        required: ['expression'],
      },
    },
  },
];

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'What is 42 * 17?' }],
  tools,
  tool_choice: 'required',
  stream: true,
});

let toolName = '';
let toolArgs = '';

for await (const chunk of stream) {
  const delta = chunk.choices[0].delta;
  if (delta.tool_calls) {
    for (const tc of delta.tool_calls) {
      if (tc.function?.name) toolName = tc.function.name;
      if (tc.function?.arguments) toolArgs += tc.function.arguments;
    }
  }
  if (chunk.choices[0].finish_reason === 'tool_calls') {
    console.log(`${toolName}(${toolArgs})`);
  }
}

ai.dispose();
```

## Context Window Management

Context overflow is handled automatically. The default strategy is `sliding_window` with a 25% reserve ratio.

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  contextSize: 2048,
});

const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
  { role: 'system', content: 'You are a helpful assistant.' },
];

for (let i = 1; i <= 20; i++) {
  messages.push({ role: 'user', content: `Question ${i}: explain concept #${i}.` });

  const response = await ai.chat.completions.create({ messages, max_tokens: 128 });
  messages.push({ role: 'assistant', content: response.choices[0].message.content! });

  const ctx = response._context!;
  if (ctx.overflowTriggered) {
    console.log(`Turn ${i}: trimmed ${ctx.originalMessageCount} → ${ctx.keptMessageCount} messages`);
  } else {
    console.log(`Turn ${i}: ${ctx.promptTokens} tokens (fits)`);
  }
}

ai.dispose();
```

## Context Overflow — Advanced Config

Full control with `reserveRatio` and the `onOverflow` callback:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  contextSize: 2048,
  contextOverflow: {
    strategy: 'sliding_window',
    reserveRatio: 0.3,
    onOverflow: (event) => {
      console.log(
        `Overflow! ${event.promptTokens} tokens > ${event.availableTokens} budget. ` +
        `Applying ${event.strategy}...`
      );
      // Optionally return replacement messages (e.g. summarized):
      // return [{ role: 'system', content: 'Summary of conversation so far...' }, ...recent];
    },
  },
});

const messages: Array<{ role: 'user' | 'assistant'; content: string }> = [];

for (let i = 1; i <= 15; i++) {
  messages.push({ role: 'user', content: `Tell me about topic #${i} in great detail.` });
  const response = await ai.chat.completions.create({ messages, max_tokens: 200 });
  messages.push({ role: 'assistant', content: response.choices[0].message.content! });
}

ai.dispose();
```

## Context Overflow — Per-Request Override

Override the strategy for a single request:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  contextSize: 2048,
});

// Normal request — uses default sliding_window
const res1 = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello' }],
});

// This request throws an error if messages don't fit
try {
  const longConversation = Array.from({ length: 100 }, (_, i) => ({
    role: 'user' as const,
    content: `Message ${i}: ${'x'.repeat(100)}`,
  }));

  await ai.chat.completions.create({
    messages: longConversation,
    context_overflow: 'error',
    max_tokens: 128,
  });
} catch (err) {
  console.log('Caught overflow error:', (err as Error).message);
}

ai.dispose();
```

## Embeddings — Basic

Generate embeddings for text. Useful for semantic search, RAG, clustering, and similarity comparison.

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  embeddings: true,
});

// Single text
const response = await ai.embeddings.create({
  input: 'What is TypeScript?',
});

console.log(response.data[0].embedding.length); // e.g. 2048
console.log(response.usage.prompt_tokens);       // token count

// Batch — multiple texts in one call (more efficient)
const batch = await ai.embeddings.create({
  input: [
    'The cat sat on the mat',
    'A kitten rested on the rug',
    'Stock prices fell sharply today',
  ],
});

console.log(`Embedded ${batch.data.length} texts`);

ai.dispose();
```

## Embeddings — Similarity Search

Use embeddings for semantic similarity. Since vectors are L2-normalized, dot product equals cosine similarity.

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  embeddings: true,
});

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

const result = await ai.embeddings.create({
  input: [
    'The cat sat on the mat',
    'A kitten rested on the rug',
    'Stock prices fell sharply today',
  ],
});

const [emb0, emb1, emb2] = result.data.map(d => d.embedding as number[]);
console.log(`"cat" vs "kitten": ${cosineSimilarity(emb0, emb1).toFixed(4)}`);  // high
console.log(`"cat" vs "stocks": ${cosineSimilarity(emb0, emb2).toFixed(4)}`);  // low

ai.dispose();
```

## Embeddings — Mini RAG Retrieval

A minimal retrieval-augmented generation (RAG) example:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  embeddings: true,
});

function dotProduct(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

const documents = [
  'TypeScript is a typed superset of JavaScript.',
  'Python is a high-level interpreted language.',
  'SQL is used for managing relational databases.',
  'Docker packages applications for consistent deployment.',
];

// Embed all documents
const docEmbs = await ai.embeddings.create({ input: documents });

// Embed the user's query
const query = 'How do I manage databases?';
const queryEmb = await ai.embeddings.create({ input: query });
const queryVec = queryEmb.data[0].embedding as number[];

// Find the most relevant document
const scored = documents.map((doc, i) => ({
  doc,
  score: dotProduct(queryVec, docEmbs.data[i].embedding as number[]),
}));
scored.sort((a, b) => b.score - a.score);

console.log(`Query: "${query}"`);
console.log(`Best match: "${scored[0].doc}" (score: ${scored[0].score.toFixed(4)})`);

// Use the retrieved document as context for the LLM
const response = await ai.chat.completions.create({
  messages: [
    { role: 'system', content: `Use this context to answer: ${scored[0].doc}` },
    { role: 'user', content: query },
  ],
  max_tokens: 256,
});

console.log(response.choices[0].message.content);

ai.dispose();
```

## Embeddings — Custom Pooling

Choose a pooling strategy for your embedding model:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: './embedding-model.gguf',
  embeddings: { poolingType: 'cls' }, // 'mean' (default), 'cls', or 'last'
});

const result = await ai.embeddings.create({
  input: 'Hello world',
});

console.log(result.data[0].embedding.length);
ai.dispose();
```

## Embeddings — Low-Level API

Use `Model` and `EmbeddingContext` directly:

```typescript
import { Model } from 'local-llm';

const model = new Model('./model.gguf');
const ctx = model.createEmbeddingContext({ poolingType: 'mean' });

console.log(`Dimension: ${ctx.dimension}`);

// Single
const vec = ctx.embed('Hello, world!');

// Batch
const vectors = ctx.embedBatch([
  'First document',
  'Second document',
  'Third document',
]);

ctx.dispose();
model.dispose();
```

## Structured Output (JSON Schema)

Force the model to output valid JSON matching a specific schema:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

// Schema-constrained JSON output
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Generate a person profile.' }],
  max_tokens: 256,
  response_format: {
    type: 'json_schema',
    json_schema: {
      name: 'person',
      schema: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          age: { type: 'integer' },
          hobbies: { type: 'array', items: { type: 'string' } },
        },
        required: ['name', 'age', 'hobbies'],
      },
    },
  },
});

const person = JSON.parse(response.choices[0].message.content);
console.log(person); // { name: "Alice", age: 30, hobbies: ["reading", "hiking"] }

ai.dispose();
```

## Free-form JSON Output

Request any valid JSON without a specific schema:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'List 3 programming languages as JSON' }],
  max_tokens: 256,
  response_format: { type: 'json_object' },
});

const data = JSON.parse(response.choices[0].message.content);
console.log(data);

ai.dispose();
```

## Custom GBNF Grammar

Use a raw GBNF grammar for full control over output format:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

// Only allow "yes" or "no" as output
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Is the sky blue?' }],
  max_tokens: 8,
  grammar: 'root ::= "yes" | "no"',
});

console.log(response.choices[0].message.content); // "yes" or "no"

ai.dispose();
```

## Low-Level Grammar with InferenceContext

```typescript
import { Model } from 'local-llm';

const model = new Model('./model.gguf', { compute: 'gpu' });
const ctx = model.createContext({ contextSize: 4096 });

const prompt = model.applyChatTemplate([
  { role: 'user', content: 'Rate this movie from 1-10 as JSON: { "rating": <number> }' },
], true);

const result = await ctx.generate(prompt, {
  maxTokens: 64,
  responseFormat: {
    type: 'json_schema',
    json_schema: {
      schema: {
        type: 'object',
        properties: { rating: { type: 'integer', minimum: 1, maximum: 10 } },
        required: ['rating'],
      },
    },
  },
});

console.log(JSON.parse(result)); // { rating: 8 }

ctx.dispose();
model.dispose();
```

## Low-Level: Tokenization

```typescript
import { Model } from 'local-llm';

const model = new Model('./model.gguf');

const tokens = model.tokenize('Hello, world!');
console.log('Token count:', tokens.length);
console.log('Token IDs:', Array.from(tokens));

const text = model.detokenize(tokens);
console.log('Decoded:', text);

model.dispose();
```

## Low-Level: Direct Inference

```typescript
import { Model } from 'local-llm';

const model = new Model('./model.gguf', { compute: 'gpu' });
const ctx = model.createContext({ contextSize: 4096 });

// Format messages with the model's chat template
const prompt = model.applyChatTemplate([
  { role: 'system', content: 'Answer briefly.' },
  { role: 'user', content: 'What is the speed of light?' },
], true);

// Generate
const result = await ctx.generate(prompt, {
  maxTokens: 128,
  temperature: 0.3,
});
console.log(result);

// Or stream
for await (const token of ctx.stream(prompt, { maxTokens: 128 })) {
  process.stdout.write(token);
}

ctx.dispose();
model.dispose();
```

## Vercel AI SDK — generateText

```typescript
import { generateText } from 'ai';
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const { text, usage } = await generateText({
  model: ai.languageModel(),
  prompt: 'What is the speed of light?',
  maxTokens: 128,
});

console.log(text);
console.log(`Tokens: ${usage.inputTokens} in, ${usage.outputTokens} out`);
ai.dispose();
```

## Vercel AI SDK — streamText

```typescript
import { streamText } from 'ai';
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const result = streamText({
  model: ai.languageModel(),
  prompt: 'Write a haiku about coding.',
  maxTokens: 64,
});

for await (const chunk of result.textStream) {
  process.stdout.write(chunk);
}
console.log();
ai.dispose();
```

## Multi-Model Pool

```typescript
import { LocalLLM } from 'local-llm';

const chat = await LocalLLM.pool.load('chat', './chat-model.gguf');
const code = await LocalLLM.pool.load('code', './code-model.gguf');

// Use models as needed — pool manages memory and eviction
const ctx = chat.createContext();
const response = await ctx.generate('Hello!', { maxTokens: 64 });
console.log(response);

ctx.dispose();
LocalLLM.pool.dispose();
```

## Performance Metrics

Every response includes `_timing` with inference speed data:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Explain quantum computing in one paragraph.' }],
  max_tokens: 256,
});

console.log(response.choices[0].message.content);
console.log(`\nPrompt eval: ${response._timing?.promptTokensPerSec.toFixed(0)} tok/s`);
console.log(`Generation:  ${response._timing?.generatedTokensPerSec.toFixed(0)} tok/s`);
console.log(`TTFT:        ${response._timing?.promptEvalMs.toFixed(0)} ms`);

ai.dispose();
```

## Streaming with Performance Metrics

When streaming, `_timing` is attached to the final chunk:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Write a haiku about code.' }],
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
  if (chunk._timing) {
    console.log(`\n\nGeneration: ${chunk._timing.generatedTokensPerSec.toFixed(1)} tok/s`);
  }
}

ai.dispose();
```

## Low-Level: Performance Metrics

Use `getPerf()` on an `InferenceContext` after any inference call:

```typescript
import { Model } from 'local-llm';

const model = new Model('./model.gguf', { compute: 'gpu' });
const ctx = model.createContext({ contextSize: 4096 });

const prompt = model.applyChatTemplate([
  { role: 'user', content: 'What is the speed of light?' },
], true);

const text = await ctx.generate(prompt, { maxTokens: 128 });
const perf = ctx.getPerf();

console.log(text);
console.log(`\nPrompt: ${perf.promptTokens} tokens in ${perf.promptEvalMs.toFixed(0)} ms (${perf.promptTokensPerSec.toFixed(0)} tok/s)`);
console.log(`Generated: ${perf.generatedTokens} tokens in ${perf.generationMs.toFixed(0)} ms (${perf.generatedTokensPerSec.toFixed(0)} tok/s)`);

ctx.dispose();
model.dispose();
```

## Benchmark

Run a reproducible benchmark to measure inference speed:

```typescript
import { Model } from 'local-llm';

const model = new Model('./model.gguf', { compute: 'gpu' });
const ctx = model.createContext({ contextSize: 4096 });

const result = await ctx.benchmark({
  promptTokens: 256,
  generateTokens: 128,
  iterations: 5,
});

console.log(`Prompt eval: ${result.promptTokensPerSec.toFixed(0)} tok/s`);
console.log(`Generation:  ${result.generatedTokensPerSec.toFixed(0)} tok/s`);
console.log(`TTFT:        ${result.ttftMs.toFixed(0)} ms`);
console.log(`Total:       ${result.totalMs.toFixed(0)} ms (${result.iterations} iterations)`);

ctx.dispose();
model.dispose();
```

## Warmup Control

By default, `LocalLLM.create()` runs a single-token warmup pass after loading the model. This primes GPU shaders and CPU caches, eliminating a 500ms–2s cold-start penalty on the first real inference.

```typescript
import { LocalLLM } from 'local-llm';

// Default: warmup runs automatically (recommended)
const ai = await LocalLLM.create({
  model: './model.gguf',
});

// First inference is fast — no cold-start penalty
const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Hello!' }],
});
```

To skip warmup (e.g. if loading many models and only using them later):

```typescript
const ai = await LocalLLM.create({
  model: './model.gguf',
  warmup: false,  // Defer context creation to first use
});
```

## Prompt Progress and Cancellation

For long prompts, you can track prompt evaluation progress and cancel mid-evaluation:

```typescript
import { Model } from 'local-llm';

const model = new Model('./model.gguf');
const ctx = model.createContext();

const prompt = model.applyChatTemplate([
  { role: 'user', content: veryLongDocument },
], true);

const text = await ctx.generate(prompt, {
  maxTokens: 256,
  onPromptProgress: (processed, total) => {
    const pct = ((processed / total) * 100).toFixed(0);
    console.log(`Evaluating prompt: ${pct}% (${processed}/${total} tokens)`);
    return true; // Return false to abort
  },
});
```

Cancel generation during prompt evaluation:

```typescript
const text = await ctx.generate(prompt, {
  maxTokens: 256,
  onPromptProgress: (processed, total) => {
    if (shouldCancel) return false; // Abort — throws cancellation error
    return true;
  },
});
```

## Speculative Decoding

Use a small draft model to predict tokens ahead, then verify in batch against the main model. Typically 2-3x faster generation with no quality loss — the output is identical to what the main model would produce.

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'meta-llama/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
  draftModel: 'meta-llama/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
});

const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Explain quantum computing in simple terms.' }],
  max_tokens: 512,
});

console.log(response.choices[0].message.content);
console.log(`Speed: ${response._timing?.generatedTokensPerSec.toFixed(0)} tok/s`);

ai.dispose();
```

Speculative decoding works with streaming too — tokens are emitted in bursts as each batch is verified:

```typescript
const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Write a poem about the sea.' }],
  max_tokens: 256,
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? '');
}
```

Tune `draftNMax` to control the draft length per step (default: 16):

```typescript
const ai = await LocalLLM.create({
  model: './main-model.gguf',
  draftModel: './draft-model.gguf',
  draftNMax: 8,  // Fewer draft tokens — less wasted compute when draft is inaccurate
});
```

## Model Management

```typescript
import { ModelManager } from 'local-llm';

const manager = new ModelManager();

// Download a model
const path = await manager.downloadModel(
  'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  {
    onProgress: (downloaded, total, pct) => {
      process.stdout.write(`\r${pct.toFixed(1)}%`);
    },
  },
);
console.log('\nSaved to:', path);

// Second call is instant (cached)
const cached = await manager.downloadModel(
  'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
);

// List all cached models
const models = await manager.listModels();
for (const m of models) {
  console.log(`${m.url} — ${(m.size / 1e9).toFixed(1)} GB`);
}

// Remove a model
await manager.removeModel(
  'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
);
```
