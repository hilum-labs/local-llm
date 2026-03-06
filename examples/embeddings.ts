/**
 * Embeddings API example — single text, batch, and similarity search.
 * Run: npx tsx examples/embeddings.ts
 */
import { LocalAI } from 'local-llm';

const ai = await LocalAI.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  embeddings: true,
});

// ── Single embedding ────────────────────────────────────────────────────────

console.log('=== Single embedding ===\n');

const response = await ai.embeddings.create({
  input: 'What is TypeScript?',
});

console.log(`Dimension: ${response.data[0].embedding.length}`);
console.log(`First 5 values: ${(response.data[0].embedding as number[]).slice(0, 5).map(v => v.toFixed(4)).join(', ')}`);
console.log(`Tokens used: ${response.usage.prompt_tokens}`);

// ── Batch embedding ─────────────────────────────────────────────────────────

console.log('\n=== Batch embedding ===\n');

const batch = await ai.embeddings.create({
  input: [
    'The cat sat on the mat',
    'A kitten rested on the rug',
    'Stock prices fell sharply today',
  ],
});

console.log(`Embedded ${batch.data.length} texts`);
console.log(`Total tokens: ${batch.usage.total_tokens}`);

// ── Cosine similarity ───────────────────────────────────────────────────────

console.log('\n=== Similarity search ===\n');

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // Already L2-normalized, so dot product = cosine similarity
}

const emb0 = batch.data[0].embedding as number[];
const emb1 = batch.data[1].embedding as number[];
const emb2 = batch.data[2].embedding as number[];

console.log(`"cat on mat" vs "kitten on rug": ${cosineSimilarity(emb0, emb1).toFixed(4)}`);
console.log(`"cat on mat" vs "stock prices":  ${cosineSimilarity(emb0, emb2).toFixed(4)}`);
console.log(`"kitten on rug" vs "stock prices": ${cosineSimilarity(emb1, emb2).toFixed(4)}`);

// ── Simple RAG retrieval ────────────────────────────────────────────────────

console.log('\n=== Mini RAG retrieval ===\n');

const documents = [
  'TypeScript is a typed superset of JavaScript that compiles to plain JavaScript.',
  'Python is a high-level, interpreted programming language known for its readability.',
  'Rust is a systems programming language focused on safety and performance.',
  'SQL is used for managing and querying relational databases.',
  'Docker containers package applications with their dependencies for consistent deployment.',
];

const docEmbeddings = await ai.embeddings.create({ input: documents });

const query = 'How do I manage databases?';
const queryEmb = await ai.embeddings.create({ input: query });
const queryVec = queryEmb.data[0].embedding as number[];

const scored = documents.map((doc, i) => ({
  doc,
  score: cosineSimilarity(queryVec, docEmbeddings.data[i].embedding as number[]),
}));

scored.sort((a, b) => b.score - a.score);

console.log(`Query: "${query}"\n`);
console.log('Top matches:');
for (const { doc, score } of scored.slice(0, 3)) {
  console.log(`  ${score.toFixed(4)} — ${doc.slice(0, 60)}...`);
}

ai.dispose();
