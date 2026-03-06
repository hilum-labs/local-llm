/**
 * Preloading example — download model at startup, use it later.
 * Run: npx tsx examples/preload.ts
 */
import { LocalAI } from 'local-llm';

const MODEL = 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';

// Start downloading immediately — app continues without blocking
console.log('Starting preload...');
const preloadPromise = LocalAI.preload(MODEL, {
  onProgress: (pct) => process.stdout.write(`\rPreloading: ${pct.toFixed(0)}%`),
});

// Simulate app doing other work while model downloads
console.log('\nApp is running, doing other things...');

// When the user needs AI, wait for preload then create (fast — already cached)
await preloadPromise;
console.log('\nPreload complete. Creating instance...');

const ai = await LocalAI.create({ model: MODEL });

const response = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'What is the speed of light?' }],
  max_tokens: 128,
});

console.log(response.choices[0].message.content);
ai.dispose();
