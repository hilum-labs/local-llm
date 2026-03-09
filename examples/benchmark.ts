/**
 * Benchmark example — measure inference speed with reproducible runs.
 * Run: npx tsx examples/benchmark.ts
 */
import { Model } from 'local-llm';

const model = await Model.load(
  'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
);
const ctx = model.createContext({ contextSize: 4096 });

console.log('Running benchmark...\n');

const result = await ctx.benchmark({
  promptTokens: 128,
  generateTokens: 64,
  iterations: 3,
});

console.log(`Prompt eval: ${result.promptTokensPerSec.toFixed(0)} tok/s`);
console.log(`Generation:  ${result.generatedTokensPerSec.toFixed(0)} tok/s`);
console.log(`TTFT:        ${result.ttftMs.toFixed(0)} ms`);
console.log(`Total:       ${result.totalMs.toFixed(0)} ms (${result.iterations} iterations)`);

console.log('\nPer-iteration breakdown:');
for (const [i, m] of result.individual.entries()) {
  console.log(`  Run ${i + 1}: prompt ${m.promptTokensPerSec.toFixed(0)} tok/s, gen ${m.generatedTokensPerSec.toFixed(0)} tok/s`);
}

ctx.dispose();
model.dispose();
