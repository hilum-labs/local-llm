import { Model } from '../packages/local-llm/src/index.js';
import type { ChatMessage } from '../packages/local-llm/src/types.js';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const MODEL_PATH = resolve(__dirname, '../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf');

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
  console.log('=== Phase 3: TypeScript API Layer ===\n');

  // 1. Load model
  const model = new Model(MODEL_PATH, { gpuLayers: 99 });
  console.log('Model loaded (TinyLlama 1.1B)');
  check('Model loaded', true);

  // 2. Create context
  const ctx = model.createContext({ contextSize: 2048 });
  console.log('InferenceContext created (n_ctx=2048)');
  check('Context created', true);

  // 3. Tokenize / detokenize round-trip
  const text = 'Hello world';
  const tokens = model.tokenize(text);
  const decoded = model.detokenize(tokens);
  console.log(`Tokenize: "${text}" → ${tokens.length} tokens → "${decoded.trim()}"`);
  check('Tokenize: "Hello world" → round-trip OK', tokens.length > 0 && decoded.includes(text));

  // 4. Chat template
  const messages: ChatMessage[] = [
    { role: 'user', content: 'Say hello in one sentence.' },
  ];
  const prompt = model.applyChatTemplate(messages, true);
  console.log(`Chat template applied (${prompt.length} chars)`);
  check('Chat template applied', prompt.length > 0 && prompt.includes('hello'));

  // 5. generate()
  const result = await ctx.generate(prompt, { maxTokens: 32, temperature: 0.7 });
  console.log(`generate(): "${result.trim().slice(0, 80)}..."`);
  check('generate()', result.length > 0);

  // 6. stream()
  process.stdout.write('stream(): ');
  let streamResult = '';
  for await (const token of ctx.stream(prompt, { maxTokens: 32, temperature: 0.7 })) {
    process.stdout.write(token);
    streamResult += token;
  }
  console.log();
  check('stream()', streamResult.length > 0);

  // 7. Dispose — no crash on double dispose
  ctx.dispose();
  ctx.dispose(); // should not throw
  model.dispose();
  model.dispose(); // should not throw
  check('Dispose: clean, no double-free', true);

  // Summary
  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
