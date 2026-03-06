/**
 * Multi-model pool example.
 * Run: npx tsx examples/multi-model.ts
 *
 * Note: Update the model paths below to point to your local GGUF files.
 */
import { LocalAI } from 'local-llm';

// Load two models into the shared pool
const chat = await LocalAI.pool.load('chat', './models/chat-model.gguf');
const code = await LocalAI.pool.load('code', './models/code-model.gguf');

// Check what's loaded
console.log('Loaded models:');
for (const info of LocalAI.pool.list()) {
  console.log(`  ${info.alias}: ${(info.sizeBytes / 1e6).toFixed(1)} MB (refs: ${info.refCount})`);
}

// Use the chat model
const chatCtx = chat.createContext({ contextSize: 2048 });
const prompt = chat.applyChatTemplate(
  [{ role: 'user', content: 'Hello, who are you?' }],
  true,
);
const response = await chatCtx.generate(prompt, { maxTokens: 64 });
console.log('\nChat response:', response);

// Clean up
chatCtx.dispose();
LocalAI.pool.dispose();
