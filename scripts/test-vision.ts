import { LocalAI } from '../packages/local-llm/src/index.js';
import { readFileSync } from 'node:fs';

let passed = 0;
let failed = 0;

function check(label: string, ok: boolean) {
  if (ok) {
    console.log(`${label.padEnd(55)} PASS`);
    passed++;
  } else {
    console.log(`${label.padEnd(55)} FAIL`);
    failed++;
  }
}

async function main() {
  console.log('=== Phase 11: Vision Integration Test ===\n');

  // Requires a vision model + projector. Example with Qwen3-VL 8B:
  //   model:     Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf
  //   projector: Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf
  //
  // Override with env vars:
  //   MODEL=path/or/url PROJECTOR=path/or/url IMAGE=path/to/image.jpg npx tsx scripts/test-vision.ts

  const modelPath = process.env.MODEL
    ?? 'Qwen/Qwen3-VL-8B-Instruct-GGUF/Qwen3VL-8B-Instruct-Q4_K_M.gguf';
  const projectorPath = process.env.PROJECTOR
    ?? 'Qwen/Qwen3-VL-8B-Instruct-GGUF/mmproj-Qwen3VL-8B-Instruct-F16.gguf';
  const imagePath = process.env.IMAGE;

  console.log(`Model:     ${modelPath}`);
  console.log(`Projector: ${projectorPath}`);
  console.log(`Image:     ${imagePath ?? '(using test base64 image)'}\n`);

  // 1. Create vision-enabled instance
  console.log('--- Init with projector ---');
  let lastPct = -1;
  const ai = await LocalAI.create({
    model: modelPath,
    projector: projectorPath,
    compute: 'auto',
    contextSize: 4096,
    onProgress: (pct) => {
      const rounded = Math.floor(pct);
      if (rounded > lastPct) {
        lastPct = rounded;
        process.stdout.write(`\rDownloading: ${rounded}%`);
      }
    },
  });
  if (lastPct >= 0) console.log();
  check('Vision model loaded with projector', true);

  // Build image content — either from file or a tiny test PNG
  let imageUrl: string;
  if (imagePath) {
    const imgBuf = readFileSync(imagePath);
    const ext = imagePath.split('.').pop()?.toLowerCase() ?? 'png';
    const mime = ext === 'jpg' || ext === 'jpeg' ? 'image/jpeg' : `image/${ext}`;
    imageUrl = `data:${mime};base64,${imgBuf.toString('base64')}`;
  } else {
    // 1x1 red PNG
    const redPng = Buffer.from(
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg==',
      'base64',
    );
    imageUrl = `data:image/png;base64,${redPng.toString('base64')}`;
  }

  // 2. Non-streaming vision
  console.log('\n--- Non-streaming vision ---');
  const response = await ai.chat.completions.create({
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: 'Describe what you see in this image briefly.' },
        { type: 'image_url', image_url: { url: imageUrl } },
      ],
    }],
    max_tokens: 128,
    temperature: 0.3,
  });
  check('Response has choices', response.choices.length > 0);
  check('Response has content', response.choices[0].message.content.length > 0);
  check('Response has usage', response.usage.total_tokens > 0);
  console.log(`  Content: "${response.choices[0].message.content.trim().slice(0, 120)}"`);

  // 3. Streaming vision
  console.log('\n--- Streaming vision ---');
  const stream = await ai.chat.completions.create({
    messages: [{
      role: 'user',
      content: [
        { type: 'text', text: 'What colors do you see?' },
        { type: 'image_url', image_url: { url: imageUrl } },
      ],
    }],
    max_tokens: 64,
    stream: true,
  });
  let streamContent = '';
  process.stdout.write('  Tokens: ');
  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) {
      process.stdout.write(content);
      streamContent += content;
    }
  }
  console.log();
  check('Stream produced content', streamContent.length > 0);

  // 4. Text-only still works (no regression)
  console.log('\n--- Text-only (regression check) ---');
  const textResponse = await ai.chat.completions.create({
    messages: [{ role: 'user', content: 'What is 2+2? Answer with just the number.' }],
    max_tokens: 16,
  });
  check('Text-only still works', textResponse.choices[0].message.content.length > 0);
  console.log(`  Content: "${textResponse.choices[0].message.content.trim()}"`);

  // Cleanup
  ai.dispose();
  check('Dispose: clean', true);

  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
