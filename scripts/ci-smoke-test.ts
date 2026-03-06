/**
 * CI Smoke Test — verifies the full inference path works end-to-end.
 * Downloads a small model, runs chat completion (non-streaming + streaming),
 * and asserts real output. Designed to run in CI after native addon build.
 *
 * Usage: npx tsx scripts/ci-smoke-test.ts [model-path]
 */
import { LocalAI } from '../packages/local-llm/src/index.js';

const MODEL_SPEC =
  process.argv[2] ??
  'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';

let passed = 0;
let failed = 0;

function assert(label: string, ok: boolean, detail?: string) {
  if (ok) {
    console.log(`  PASS  ${label}`);
    passed++;
  } else {
    console.log(`  FAIL  ${label}${detail ? ` — ${detail}` : ''}`);
    failed++;
  }
}

async function main() {
  console.log('=== CI Smoke Test ===\n');
  console.log(`Model: ${MODEL_SPEC}\n`);

  // 1. Create + init
  console.log('--- Init ---');
  let lastPct = -1;
  const ai = await LocalAI.create({
    model: MODEL_SPEC,
    compute: 'auto',
    contextSize: 2048,
    onProgress: (pct) => {
      const rounded = Math.floor(pct);
      if (rounded % 10 === 0 && rounded > lastPct) {
        lastPct = rounded;
        process.stdout.write(`\r  Downloading: ${rounded}%`);
      }
    },
  });
  if (lastPct >= 0) console.log();
  assert('LocalAI.create() succeeded', true);

  // 2. Non-streaming chat completion
  console.log('\n--- Non-streaming completion ---');
  const response = await ai.chat.completions.create({
    messages: [{ role: 'user', content: 'Reply with exactly: hello world' }],
    max_tokens: 32,
    temperature: 0.1,
  });

  const content = response.choices[0]?.message?.content ?? '';
  assert('Response has choices', response.choices.length > 0);
  assert('Response content is non-empty', content.length > 0, `got "${content.slice(0, 60)}"`);
  assert('Response has usage stats', response.usage?.total_tokens > 0);
  assert('Response shape is OpenAI-compatible', response.object === 'chat.completion');

  // 3. Streaming chat completion
  console.log('\n--- Streaming completion ---');
  const stream = await ai.chat.completions.create({
    messages: [{ role: 'user', content: 'Say hi' }],
    max_tokens: 16,
    stream: true,
  });

  let streamContent = '';
  let chunkCount = 0;
  for await (const chunk of stream) {
    const delta = chunk.choices[0]?.delta?.content;
    if (delta) streamContent += delta;
    chunkCount++;
  }

  assert('Stream produced chunks', chunkCount > 0, `${chunkCount} chunks`);
  assert('Stream content is non-empty', streamContent.length > 0, `got "${streamContent.slice(0, 60)}"`);

  // 4. Dispose
  console.log('\n--- Cleanup ---');
  ai.dispose();
  assert('Dispose completed without error', true);

  // Summary
  console.log(`\n=== ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('\nFatal:', err);
  process.exit(1);
});
