/**
 * Vercel AI SDK integration example (generateText + streamText).
 * Requires: npm install ai
 * Run: npx tsx examples/ai-sdk-integration.ts
 */
import { generateText, streamText } from 'ai';
import { LocalAI } from 'local-llm';

const ai = await LocalAI.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

// --- generateText ---
console.log('=== generateText ===');
const { text, usage } = await generateText({
  model: ai.languageModel(),
  prompt: 'What is the speed of light?',
  maxTokens: 128,
});
console.log(text);
console.log(`Tokens: ${usage.inputTokens} in, ${usage.outputTokens} out\n`);

// --- streamText ---
console.log('=== streamText ===');
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
