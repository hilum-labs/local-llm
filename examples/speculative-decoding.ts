/**
 * Speculative decoding example — use a draft model for 2-3x faster generation.
 * Run: npx tsx examples/speculative-decoding.ts
 *
 * Requirements: two models from the same family (same tokenizer).
 * Example: Llama 3.2 3B (main) + Llama 3.2 1B (draft)
 */
import { LocalLLM } from 'local-llm';

// ── With speculative decoding ────────────────────────────────────────────────

console.log('Loading with speculative decoding...\n');

const ai = await LocalLLM.create({
  model: 'meta-llama/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf',
  draftModel: 'meta-llama/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf',
});

const t0 = performance.now();
const response = await ai.chat.completions.create({
  messages: [
    { role: 'system', content: 'You are a helpful assistant.' },
    { role: 'user', content: 'Explain how a CPU works in 3 paragraphs.' },
  ],
  max_tokens: 512,
});

const elapsed = performance.now() - t0;

console.log(response.choices[0].message.content);
console.log(`\n--- Speculative decoding ---`);
console.log(`Time: ${elapsed.toFixed(0)} ms`);
console.log(`Speed: ${response._timing?.generatedTokensPerSec.toFixed(0)} tok/s`);
console.log(`Tokens: ${response._timing?.generatedTokens} generated`);

// ── Streaming with speculative decoding ──────────────────────────────────────

console.log('\n--- Streaming ---\n');

const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Write a haiku about AI.' }],
  max_tokens: 64,
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
  if (chunk._timing) {
    console.log(`\n\nSpeed: ${chunk._timing.generatedTokensPerSec.toFixed(0)} tok/s`);
  }
}

ai.dispose();
