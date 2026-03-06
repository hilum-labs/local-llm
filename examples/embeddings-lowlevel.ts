/**
 * Low-level embeddings using Model + EmbeddingContext directly.
 * Run: npx tsx examples/embeddings-lowlevel.ts
 */
import { Model } from 'local-llm';

const model = new Model('./model.gguf');

console.log(`Embedding dimension: ${model.embeddingDimension}`);

const ctx = model.createEmbeddingContext({
  poolingType: 'mean',
});

// Single embedding
const vec = ctx.embed('Hello, world!');
console.log(`Vector length: ${vec.length}`);
console.log(`First 5: ${Array.from(vec.slice(0, 5)).map(v => v.toFixed(4)).join(', ')}`);

// Batch embedding
const vectors = ctx.embedBatch([
  'First document about programming',
  'Second document about cooking',
  'Third document about programming languages',
]);

console.log(`\nBatch: ${vectors.length} embeddings`);

// Cosine similarity (vectors are already L2-normalized)
function dotProduct(a: Float32Array, b: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

console.log(`\nSimilarities:`);
console.log(`  doc0 vs doc1: ${dotProduct(vectors[0], vectors[1]).toFixed(4)}`);
console.log(`  doc0 vs doc2: ${dotProduct(vectors[0], vectors[2]).toFixed(4)}`);

ctx.dispose();
model.dispose();
