/**
 * Basic chat completion example.
 * Run: npx tsx examples/basic-chat.ts
 */
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const response = await ai.chat.completions.create({
  messages: [
    { role: 'system', content: 'You are a helpful assistant. Keep answers brief.' },
    { role: 'user', content: 'What is the capital of France?' },
  ],
  max_tokens: 128,
  temperature: 0.7,
});

console.log(response.choices[0].message.content);
console.log(`\nTokens: ${response.usage.prompt_tokens} prompt, ${response.usage.completion_tokens} completion`);

ai.dispose();
