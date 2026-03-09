/**
 * Streaming chat completion example.
 * Run: npx tsx examples/streaming.ts
 */
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
