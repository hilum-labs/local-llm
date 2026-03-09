/**
 * Vision / multimodal example — send images alongside text.
 * Requires a vision model + projector (e.g. Qwen3-VL).
 * Run: npx tsx examples/vision.ts
 */
import { LocalLLM } from 'local-llm';
import { readFileSync } from 'node:fs';

const ai = await LocalLLM.create({
  model: 'Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf',
  projector: 'Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf',
});

// Option 1: base64 data URI
const imageBuffer = readFileSync('./photo.jpg');
const dataUri = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`;

const response = await ai.chat.completions.create({
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: 'Describe what you see in this image.' },
      { type: 'image_url', image_url: { url: dataUri } },
    ],
  }],
  max_tokens: 256,
});

console.log(response.choices[0].message.content);

// Option 2: local file path
const response2 = await ai.chat.completions.create({
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: 'What colors are in this image?' },
      { type: 'image_url', image_url: { url: './photo.jpg' } },
    ],
  }],
  max_tokens: 128,
});

console.log(response2.choices[0].message.content);

// Option 3: streaming with vision
const stream = await ai.chat.completions.create({
  messages: [{
    role: 'user',
    content: [
      { type: 'text', text: 'What is happening in this photo?' },
      { type: 'image_url', image_url: { url: './photo.jpg' } },
    ],
  }],
  max_tokens: 256,
  stream: true,
});

for await (const chunk of stream) {
  const content = chunk.choices[0]?.delta?.content;
  if (content) process.stdout.write(content);
}
console.log();

ai.dispose();
