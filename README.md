[![npm](https://img.shields.io/npm/v/local-llm)](https://www.npmjs.com/package/local-llm)
[![npm downloads](https://img.shields.io/npm/dm/local-llm)](https://www.npmjs.com/package/local-llm)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![Platform: macOS | Linux | Windows](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20Windows-lightgrey)

# local-llm

Run LLMs locally in Node.js with an OpenAI-compatible API. No cloud, no API keys, no data leaves your machine.

```bash
npm install local-llm
```

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'What is the capital of France?' }],
  max_tokens: 128,
});

console.log(response.choices[0].message.content);
```

> Need to run on **mobile**? Check out [`local-llm-rn`](https://www.npmjs.com/package/local-llm-rn) for React Native with Metal (iOS) and Vulkan (Android) GPU acceleration.

## Why local-llm?

- **Not a server.** Unlike Ollama, there's no daemon to run. It's just an npm package.
- **OpenAI-compatible out of the box.** Unlike node-llama-cpp, you get `chat.completions.create()` with zero boilerplate.
- **Your data stays local.** Unlike cloud APIs, nothing leaves your machine. No API keys, no usage limits, no latency.
- **One install.** Native C++ bindings compile automatically. No Python, no Docker, no external processes.

## Features

- **OpenAI-compatible API** - Same `chat.completions.create()` interface you already know
- **Vision / Multimodal** - Send images alongside text using the GPT-4V content format
- **Vercel AI SDK** - Drop-in provider for `generateText()` and `streamText()`
- **Auto model download** - Pass a HuggingFace URL or shorthand, models are downloaded and cached automatically
- **GPU auto-detection** - Detects Metal (macOS) and CUDA (Linux/Windows) automatically
- **Streaming** - Full streaming support via async iterators
- **TypeScript-first** - Complete type definitions out of the box
- **No dependencies** - Native C++ bindings to llama.cpp, no Python, no external servers
- **Fast** - ~80 tok/s generation on M2 MacBook Pro with Llama 3.2 3B Q4_K_M
- **Speculative decoding** - Use a small draft model for 2-3x faster generation with zero quality loss

## Platform Support

| Platform | GPU | Status |
|---|---|---|
| macOS Apple Silicon (M1-M4) | Metal | Supported |
| macOS Intel | Metal | Supported |
| Linux x64 | CPU | Supported |
| Windows x64 | CPU | Supported |
| Linux ARM64 | CPU | Coming soon |
| Linux/Windows CUDA | NVIDIA GPU | Coming soon |

## Quick Start

### 1. Install

```bash
npm install local-llm
```

### 2. Choose a Model

Any GGUF model from HuggingFace works. Some recommendations:

| Model | Size | Good for |
|---|---|---|
| [TinyLlama 1.1B Q4_K_M](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) | ~636 MB | Testing, development |
| [Llama 3.2 3B Q4_K_M](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF) | ~1.8 GB | Fast, great quality |
| [Phi-3 Mini Q4_K_M](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) | ~2.2 GB | Lightweight, fast |
| [Llama 3.1 8B Q4_K_M](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF) | ~4.9 GB | Best quality |
| [Mistral 7B Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) | ~4.4 GB | General use |

### 3. Use

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

// Chat completion (same API as OpenAI)
const response = await ai.chat.completions.create({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Explain gravity in one sentence.' },
  ],
  max_tokens: 128,
  temperature: 0.7,
});

console.log(response.choices[0].message.content);

// Streaming
const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Write a haiku about coding.' }],
  stream: true,
});

for await (const chunk of stream) {
  process.stdout.write(chunk.choices[0]?.delta?.content ?? '');
}

// Clean up
ai.dispose();
```

### Vercel AI SDK

```typescript
import { generateText } from 'ai';
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({ model: 'user/repo/model.gguf' });
const { text } = await generateText({ model: ai.languageModel(), prompt: 'Hello!' });
console.log(text);
ai.dispose();
```

### Preloading

Pre-download a model at app startup so users don't wait:

```typescript
// App startup — download runs in the background, app doesn't block
LocalLLM.preload('user/repo/model.gguf');

// Later, when AI is needed — cached, create() is fast
const ai = await LocalLLM.create({ model: 'user/repo/model.gguf' });
```

### Vision / Multimodal

Send images alongside text using the same OpenAI GPT-4V content format. Requires a vision model and its projector file:

```typescript
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf',
  projector: 'Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf',
});

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
ai.dispose();
```

Images can be provided as `data:` URIs (base64), local file paths, or HTTP URLs. Streaming works too — just add `stream: true`.

## Configuration

```typescript
const ai = await LocalLLM.create({
  // Model source (required)
  model: 'user/repo/file.gguf',       // HuggingFace shorthand
  // model: 'https://huggingface.co/...', // Full URL
  // model: './models/my-model.gguf',     // Local file path

  // Vision projector (optional — required for vision models)
  // projector: 'user/repo/mmproj-file.gguf',

  // Compute mode (default: 'auto')
  compute: 'auto',    // Auto-detect GPU
  // compute: 'gpu',  // Force GPU (Metal/CUDA)
  // compute: 'cpu',  // Force CPU only
  // compute: 'hybrid', // Split between CPU and GPU

  // Context options
  contextSize: 2048,   // Context window size
  batchSize: 512,      // Batch size for prompt processing
  threads: 4,          // CPU thread count

  // Performance
  warmup: true,          // Warmup on load (eliminates cold-start). Default: true

  // Speculative decoding (optional — 2-3x faster generation)
  // draftModel: 'user/repo/small-model.gguf',  // Small model from same family
  // draftNMax: 16,                              // Max draft tokens per step

  // Download options
  cacheDir: '~/.local-llm/models',  // Model cache directory
  onProgress: (pct) => {            // Download progress callback
    console.log(`${pct.toFixed(1)}%`);
  },
});
```

## Generation Options

```typescript
const response = await ai.chat.completions.create({
  messages: [...],
  max_tokens: 256,       // Maximum tokens to generate
  temperature: 0.7,      // Randomness (0.0 = deterministic, 2.0 = very random)
  top_p: 0.9,            // Nucleus sampling
  top_k: 40,             // Top-k sampling
  frequency_penalty: 1.1, // Repetition penalty
  seed: 42,              // Reproducible output
  stream: false,         // Set to true for streaming
});
```

## Peer Dependencies

The Vercel AI SDK integration is optional. Install `ai` if you want to use `generateText()` / `streamText()`:

```bash
npm install ai
```

## API Reference

See the full [API documentation](docs/api.md).

## Advanced Usage

For lower-level control, you can use the engine classes directly:

```typescript
import { Model, InferenceContext } from 'local-llm';

const model = new Model('./model.gguf', { compute: 'gpu' });
const ctx = model.createContext({ contextSize: 4096 });

// Tokenize
const tokens = model.tokenize('Hello world');
const text = model.detokenize(tokens);

// Chat template
const prompt = model.applyChatTemplate([
  { role: 'user', content: 'Hello' },
], true);

// Generate
const result = await ctx.generate(prompt, { maxTokens: 128 });

// Stream
for await (const token of ctx.stream(prompt, { maxTokens: 128 })) {
  process.stdout.write(token);
}

ctx.dispose();
model.dispose();
```

## Model Manager

Download and cache models programmatically:

```typescript
import { ModelManager } from 'local-llm';

const manager = new ModelManager();

// Download with progress
const path = await manager.downloadModel(
  'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  {
    onProgress: (downloaded, total, pct) => {
      console.log(`${pct.toFixed(1)}%`);
    },
  },
);

// List cached models
const models = await manager.listModels();

// Remove a cached model
await manager.removeModel('https://huggingface.co/...');
```

## Ecosystem

| Package | Description | Install |
|---|---|---|
| [`local-llm`](https://www.npmjs.com/package/local-llm) | Node.js / Bun / Electron (this package) | `npm install local-llm` |
| [`local-llm-rn`](https://github.com/hilum-labs/local-llm-rn) | React Native / Expo (iOS Metal, Android Vulkan) | `npm install local-llm-rn` |
| [`hilum-local-llm-engine`](https://github.com/hilum-labs/hilum-local-llm-engine) | Core C++ engine (llama.cpp fork) | Vendored automatically |

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

Questions, feedback, or partnership inquiries: [info@hilumlabs.com](mailto:info@hilumlabs.com)

## License

MIT - See [LICENSE](LICENSE) for details.

Made by [Hilum Labs](https://github.com/hilum-labs).
