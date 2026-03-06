/**
 * Function / tool calling example.
 * The model decides which tool to call and generates structured arguments.
 * Run: npx tsx examples/tool-calling.ts
 */
import { LocalAI } from 'local-llm';
import type { ChatCompletionTool, ChatCompletionRequestMessage } from 'local-llm';

const MODEL = 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';

const tools: ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'get_weather',
      description: 'Get the current weather for a location',
      parameters: {
        type: 'object',
        properties: {
          location: { type: 'string', description: 'City name, e.g. "Paris"' },
          unit: { type: 'string', enum: ['celsius', 'fahrenheit'] },
        },
        required: ['location'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'search_web',
      description: 'Search the web for information',
      parameters: {
        type: 'object',
        properties: {
          query: { type: 'string', description: 'Search query' },
        },
        required: ['query'],
      },
    },
  },
];

// Fake tool implementations
function getWeather(args: { location: string; unit?: string }) {
  return { temperature: 22, condition: 'sunny', location: args.location, unit: args.unit ?? 'celsius' };
}

function searchWeb(args: { query: string }) {
  return { results: [`Top result for "${args.query}": Example page content...`] };
}

const ai = await LocalAI.create({ model: MODEL });

// ── Auto mode: model decides whether to call a tool ─────────────────────────

console.log('=== Auto mode (model decides) ===\n');

const res1 = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'What is the weather in Paris?' }],
  tools,
  tool_choice: 'auto',
  max_tokens: 256,
});

if (res1.choices[0].finish_reason === 'tool_calls') {
  const call = res1.choices[0].message.tool_calls![0];
  console.log(`Tool called: ${call.function.name}(${call.function.arguments})`);
} else {
  console.log(`Text response: ${res1.choices[0].message.content}`);
}

// ── Required mode: model must call a tool (grammar-constrained) ─────────────

console.log('\n=== Required mode (grammar-constrained) ===\n');

const res2 = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Look up the weather in Tokyo' }],
  tools,
  tool_choice: 'required',
  max_tokens: 256,
});

const toolCall = res2.choices[0].message.tool_calls![0];
console.log(`Tool: ${toolCall.function.name}`);
console.log(`Args: ${toolCall.function.arguments}`);

// ── Specific function: force a particular tool ──────────────────────────────

console.log('\n=== Specific function (force get_weather) ===\n');

const res3 = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'Tell me about New York' }],
  tools,
  tool_choice: { type: 'function', function: { name: 'get_weather' } },
  max_tokens: 256,
});

const forced = res3.choices[0].message.tool_calls![0];
console.log(`Forced tool: ${forced.function.name}(${forced.function.arguments})`);

// ── Multi-turn: execute tool and send result back ───────────────────────────

console.log('\n=== Multi-turn tool use ===\n');

const messages: ChatCompletionRequestMessage[] = [
  { role: 'user', content: 'What is the weather in London right now?' },
];

const turn1 = await ai.chat.completions.create({
  messages,
  tools,
  tool_choice: 'required',
  max_tokens: 256,
});

const call = turn1.choices[0].message.tool_calls![0];
console.log(`Model wants to call: ${call.function.name}(${call.function.arguments})`);

const args = JSON.parse(call.function.arguments);
const toolResult = call.function.name === 'get_weather'
  ? getWeather(args)
  : searchWeb(args);

console.log(`Tool result: ${JSON.stringify(toolResult)}`);

messages.push(
  { role: 'assistant', content: null, tool_calls: turn1.choices[0].message.tool_calls },
  { role: 'tool', content: JSON.stringify(toolResult), tool_call_id: call.id, name: call.function.name },
);

const turn2 = await ai.chat.completions.create({
  messages,
  tools,
  max_tokens: 256,
});

console.log(`\nFinal answer: ${turn2.choices[0].message.content}`);

ai.dispose();
