/**
 * Warmup example — compare cold-start vs warmed-up inference latency.
 * Run: npx tsx examples/warmup.ts
 */
import { LocalLLM } from 'local-llm';

const messages = [{ role: 'user' as const, content: 'Say hello in one word.' }];

// ── With warmup (default) — first inference has no cold-start penalty ────────

console.log('Loading model with warmup (default)...');
const t0 = performance.now();
const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  // warmup: true  (this is the default)
});
console.log(`Model ready in ${(performance.now() - t0).toFixed(0)} ms (includes warmup)\n`);

const t1 = performance.now();
const r1 = await ai.chat.completions.create({ messages, max_tokens: 16 });
console.log(`First inference: ${(performance.now() - t1).toFixed(0)} ms — "${r1.choices[0].message.content}"`);

const t2 = performance.now();
const r2 = await ai.chat.completions.create({ messages, max_tokens: 16 });
console.log(`Second inference: ${(performance.now() - t2).toFixed(0)} ms — "${r2.choices[0].message.content}"`);

ai.dispose();

// ── Without warmup — first inference pays the cold-start cost ────────────────

console.log('\nLoading model without warmup...');
const t3 = performance.now();
const ai2 = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  warmup: false,
});
console.log(`Model ready in ${(performance.now() - t3).toFixed(0)} ms (no warmup)\n`);

const t4 = performance.now();
const r3 = await ai2.chat.completions.create({ messages, max_tokens: 16 });
console.log(`First inference: ${(performance.now() - t4).toFixed(0)} ms — "${r3.choices[0].message.content}" (cold-start included)`);

const t5 = performance.now();
const r4 = await ai2.chat.completions.create({ messages, max_tokens: 16 });
console.log(`Second inference: ${(performance.now() - t5).toFixed(0)} ms — "${r4.choices[0].message.content}"`);

ai2.dispose();
