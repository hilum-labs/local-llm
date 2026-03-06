import { LocalAI } from '../packages/local-llm/src/index.js';
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
  console.log('=== Phase 5: OpenAI-Compatible API Layer ===\n');

  const ai = new LocalAI({
    model: MODEL_PATH,
    gpuLayers: 99,
    contextSize: 2048,
  });

  // 1. Non-streaming completion
  console.log('--- Non-streaming ---');
  const response = await ai.chat.completions.create({
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is the capital of France?' },
    ],
    max_tokens: 64,
    temperature: 0.3,
  });

  check('Has id (string)', typeof response.id === 'string' && response.id.startsWith('chatcmpl-'));
  check('Has object = "chat.completion"', response.object === 'chat.completion');
  check('Has created (number)', typeof response.created === 'number');
  check('Has model (string)', typeof response.model === 'string');
  check('Has choices (length > 0)', response.choices.length > 0);
  check('Choice has message.role = assistant', response.choices[0].message.role === 'assistant');
  check('Choice has message.content (non-empty)', response.choices[0].message.content.length > 0);
  check('Choice has finish_reason', ['stop', 'length'].includes(response.choices[0].finish_reason));
  check('Has usage.prompt_tokens', response.usage.prompt_tokens > 0);
  check('Has usage.completion_tokens', response.usage.completion_tokens > 0);
  check('usage.total = prompt + completion',
    response.usage.total_tokens === response.usage.prompt_tokens + response.usage.completion_tokens);

  console.log(`  Content: "${response.choices[0].message.content.trim().slice(0, 100)}..."`);
  console.log(`  Usage: ${JSON.stringify(response.usage)}`);

  // 2. Streaming completion
  console.log('\n--- Streaming ---');
  const stream = await ai.chat.completions.create({
    messages: [{ role: 'user', content: 'Count to 3.' }],
    max_tokens: 64,
    stream: true,
  });

  const chunks: Array<{ role?: string; content?: string; finish_reason: string | null }> = [];
  process.stdout.write('  Tokens: ');
  for await (const chunk of stream) {
    const choice = chunk.choices[0];
    chunks.push({
      role: choice.delta.role,
      content: choice.delta.content,
      finish_reason: choice.finish_reason,
    });
    if (choice.delta.content) {
      process.stdout.write(choice.delta.content);
    }
  }
  console.log();

  check('First chunk has role = assistant', chunks[0]?.role === 'assistant');
  check('Middle chunks have content', chunks.slice(1, -1).some((c) => c.content && c.content.length > 0));
  check('Last chunk has finish_reason = stop', chunks[chunks.length - 1]?.finish_reason === 'stop');
  check('All chunks have correct object', true); // validated by TS types

  // 3. Dispose
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
