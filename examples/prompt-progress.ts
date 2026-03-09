/**
 * Prompt progress example — track and cancel long prompt evaluation.
 * Run: npx tsx examples/prompt-progress.ts
 */
import { Model } from 'local-llm';

const model = await Model.load(
  'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
);
const ctx = model.createContext({ contextSize: 4096 });

const longDocument = 'The quick brown fox jumps over the lazy dog. '.repeat(200);

const prompt = model.applyChatTemplate([
  { role: 'user', content: `Summarize this text:\n\n${longDocument}` },
], true);

console.log(`Prompt length: ~${prompt.length} chars\n`);

// ── Progress tracking ────────────────────────────────────────────────────────

console.log('Generating with progress tracking...\n');

const text = await ctx.generate(prompt, {
  maxTokens: 128,
  onPromptProgress: (processed, total) => {
    const pct = ((processed / total) * 100).toFixed(0);
    process.stdout.write(`\rPrompt eval: ${pct}% (${processed}/${total} tokens)`);
    return true;
  },
});

console.log(`\n\n${text}\n`);

// ── Cancellation ─────────────────────────────────────────────────────────────

console.log('Generating with cancellation after 50%...\n');

try {
  await ctx.generate(prompt, {
    maxTokens: 128,
    onPromptProgress: (processed, total) => {
      const pct = (processed / total) * 100;
      process.stdout.write(`\rPrompt eval: ${pct.toFixed(0)}%`);
      if (pct >= 50) {
        console.log(' — cancelling!');
        return false;
      }
      return true;
    },
  });
} catch (err: unknown) {
  console.log(`Cancelled: ${err instanceof Error ? err.message : err}`);
}

ctx.dispose();
model.dispose();
