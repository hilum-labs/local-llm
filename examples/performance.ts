/**
 * Performance metrics example — shows inference speed in responses.
 * Run: npx tsx examples/performance.ts
 */
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

// ── Non-streaming: _timing on the response ──────────────────────────────────

const response = await ai.chat.completions.create({
  messages: [
    { role: 'system', content: 'You are a helpful assistant. Keep answers brief.' },
    { role: 'user', content: 'Explain quantum computing in one sentence.' },
  ],
  max_tokens: 128,
});

console.log(response.choices[0].message.content);
console.log(`\nPrompt eval: ${response._timing?.promptTokensPerSec.toFixed(0)} tok/s (${response._timing?.promptEvalMs.toFixed(0)} ms)`);
console.log(`Generation:  ${response._timing?.generatedTokensPerSec.toFixed(0)} tok/s (${response._timing?.generationMs.toFixed(0)} ms)`);
console.log(`Tokens: ${response._timing?.promptTokens} prompt, ${response._timing?.generatedTokens} generated`);

// ── Streaming: _timing on the final chunk ───────────────────────────────────

console.log('\n--- Streaming ---\n');

const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Write a haiku about code.' }],
  max_tokens: 64,
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
  if (chunk._timing) {
    console.log(`\n\nGeneration: ${chunk._timing.generatedTokensPerSec.toFixed(1)} tok/s`);
    console.log(`TTFT: ${chunk._timing.promptEvalMs.toFixed(0)} ms`);
  }
}

ai.dispose();
