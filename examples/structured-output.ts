/**
 * Structured output examples: JSON Schema, free JSON, and GBNF grammar.
 * Run: npx tsx examples/structured-output.ts
 */
import { LocalLLM } from 'local-llm';

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

// ── JSON Schema: strict typed output ────────────────────────────────────────

console.log('=== JSON Schema output ===\n');

const schemaResponse = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Generate a fictional product listing.' }],
  max_tokens: 256,
  response_format: {
    type: 'json_schema',
    json_schema: {
      name: 'product',
      schema: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          price: { type: 'number' },
          category: { type: 'string', enum: ['electronics', 'clothing', 'food', 'books'] },
          in_stock: { type: 'boolean' },
          tags: { type: 'array', items: { type: 'string' } },
        },
        required: ['name', 'price', 'category', 'in_stock'],
      },
    },
  },
});

const product = JSON.parse(schemaResponse.choices[0].message.content!);
console.log('Product:', product);
console.log(`Type check — name is string: ${typeof product.name === 'string'}`);
console.log(`Type check — price is number: ${typeof product.price === 'number'}`);

// ── Free-form JSON ──────────────────────────────────────────────────────────

console.log('\n=== Free-form JSON ===\n');

const jsonResponse = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'List 3 famous scientists as JSON with name and field.' }],
  max_tokens: 256,
  response_format: { type: 'json_object' },
});

console.log('JSON output:', jsonResponse.choices[0].message.content);

// ── GBNF grammar: yes/no classifier ────────────────────────────────────────

console.log('\n=== GBNF grammar (yes/no) ===\n');

const questions = [
  'Is the Earth round?',
  'Is water dry?',
  'Can fish swim?',
];

for (const q of questions) {
  const res = await ai.chat.completions.create({
    messages: [{ role: 'user', content: q }],
    max_tokens: 8,
    grammar: 'root ::= "yes" | "no"',
  });
  console.log(`Q: ${q} → A: ${res.choices[0].message.content}`);
}

// ── GBNF grammar: sentiment analysis ────────────────────────────────────────

console.log('\n=== GBNF grammar (sentiment) ===\n');

const sentimentGrammar = 'root ::= "positive" | "negative" | "neutral"';

const reviews = [
  'This product is amazing, I love it!',
  'Terrible quality, broke after one day.',
  'It works fine, nothing special.',
];

for (const review of reviews) {
  const res = await ai.chat.completions.create({
    messages: [
      { role: 'system', content: 'Classify the sentiment of the following text.' },
      { role: 'user', content: review },
    ],
    max_tokens: 8,
    grammar: sentimentGrammar,
  });
  console.log(`"${review.slice(0, 40)}..." → ${res.choices[0].message.content}`);
}

ai.dispose();
