import { LocalLLM } from '../packages/local-llm/src/index.js';

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
  console.log('=== Phase 7: Full End-to-End Test ===\n');

  // 1. Create with HuggingFace URL + auto-download
  console.log('--- Init with HuggingFace URL ---');
  let lastPct = -1;
  const ai = new LocalLLM({
    model: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    compute: 'gpu',
    contextSize: 2048,
    onProgress: (pct) => {
      const rounded = Math.floor(pct);
      if (rounded > lastPct) {
        lastPct = rounded;
        process.stdout.write(`\rDownloading model: ${rounded}%`);
      }
    },
  });

  await ai.init();
  if (lastPct >= 0) console.log(); // newline after progress
  check('init() completed', true);

  // 2. Non-streaming via chat.completions
  console.log('\n--- Non-streaming ---');
  const response = await ai.chat.completions.create({
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is 2+2?' },
    ],
    max_tokens: 32,
    temperature: 0.3,
  });
  check('Response has choices', response.choices.length > 0);
  check('Response has content', response.choices[0].message.content.length > 0);
  check('Response has usage', response.usage.total_tokens > 0);
  console.log(`  Content: "${response.choices[0].message.content.trim().slice(0, 80)}"`);

  // 3. Streaming
  console.log('\n--- Streaming ---');
  const stream = await ai.chat.completions.create({
    messages: [{ role: 'user', content: 'Say hello briefly.' }],
    max_tokens: 32,
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

  // 4. Static factory method
  console.log('\n--- LocalLLM.create() ---');
  const ai2 = await LocalLLM.create({
    model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    compute: 'gpu',
  });
  const r2 = await ai2.chat.completions.create({
    messages: [{ role: 'user', content: 'Hi' }],
    max_tokens: 8,
  });
  check('create() + generate works', r2.choices[0].message.content.length > 0);
  ai2.dispose();

  // 5. Compute modes
  console.log('\n--- Compute: CPU mode ---');
  const aiCpu = await LocalLLM.create({
    model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    compute: 'cpu',
  });
  const cpuResult = await aiCpu.chat.completions.create({
    messages: [{ role: 'user', content: 'Hi' }],
    max_tokens: 8,
  });
  check('CPU mode generates output', cpuResult.choices[0].message.content.length > 0);
  aiCpu.dispose();

  // 6. Auto compute detection
  console.log('\n--- Compute: auto mode ---');
  const aiAuto = await LocalLLM.create({
    model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  });
  const autoResult = await aiAuto.chat.completions.create({
    messages: [{ role: 'user', content: 'Hi' }],
    max_tokens: 8,
  });
  check('Auto mode detects backend + generates', autoResult.choices[0].message.content.length > 0);
  aiAuto.dispose();

  // 7. Cleanup
  ai.dispose();
  check('Dispose: clean', true);

  // Summary
  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
