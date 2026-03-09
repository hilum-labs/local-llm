/**
 * Context window management example.
 * Demonstrates auto-detection, sliding window, and the onOverflow callback.
 * Run: npx tsx examples/context-overflow.ts
 */
import { LocalLLM } from 'local-llm';

// ── Basic: defaults just work (sliding_window, 25% reserve) ─────────────────

console.log('=== Default context management ===\n');

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  contextSize: 2048,
});

const messages: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
  { role: 'system', content: 'You are a helpful assistant. Keep answers brief.' },
];

for (let i = 1; i <= 20; i++) {
  messages.push({ role: 'user', content: `Question ${i}: explain concept number ${i} in software engineering.` });

  const response = await ai.chat.completions.create({ messages, max_tokens: 128 });
  const reply = response.choices[0].message.content!;
  messages.push({ role: 'assistant', content: reply });

  const ctx = response._context!;
  const status = ctx.overflowTriggered
    ? `TRIMMED (${ctx.originalMessageCount} → ${ctx.keptMessageCount} msgs, strategy: ${ctx.strategyUsed})`
    : `OK (${ctx.promptTokens} tokens)`;

  console.log(`Turn ${i}: ${status}`);
}

ai.dispose();

// ── Advanced: custom config with onOverflow callback ────────────────────────

console.log('\n=== Advanced config with onOverflow ===\n');

const ai2 = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  contextSize: 2048,
  contextOverflow: {
    strategy: 'sliding_window',
    reserveRatio: 0.3,
    onOverflow: (event) => {
      console.log(
        `  [overflow] ${event.promptTokens} tokens > ${event.availableTokens} budget ` +
        `(context: ${event.contextSize}, reserved: ${event.reservedForGeneration})`
      );
    },
  },
});

const msgs2: Array<{ role: 'system' | 'user' | 'assistant'; content: string }> = [
  { role: 'system', content: 'You are a coding tutor.' },
];

for (let i = 1; i <= 10; i++) {
  msgs2.push({ role: 'user', content: `Tell me about design pattern #${i}. Include a code example.` });

  const response = await ai2.chat.completions.create({ messages: msgs2, max_tokens: 200 });
  const reply = response.choices[0].message.content!;
  msgs2.push({ role: 'assistant', content: reply });

  console.log(`Turn ${i}: kept ${response._context!.keptMessageCount}/${response._context!.originalMessageCount} messages`);
}

ai2.dispose();

// ── Per-request override ────────────────────────────────────────────────────

console.log('\n=== Per-request strategy override ===\n');

const ai3 = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  contextSize: 2048,
  contextOverflow: 'sliding_window',
});

try {
  const longConversation = Array.from({ length: 100 }, (_, i) => ({
    role: 'user' as const,
    content: `Message ${i}: ${'x'.repeat(100)}`,
  }));

  await ai3.chat.completions.create({
    messages: longConversation,
    context_overflow: 'error',
    max_tokens: 128,
  });
} catch (err) {
  console.log(`Caught expected error: ${(err as Error).message.slice(0, 100)}...`);
}

ai3.dispose();
