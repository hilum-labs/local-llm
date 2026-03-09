/**
 * Streaming tool calling example.
 * When tools are active, the stream buffers internally and emits
 * OpenAI-compatible tool_calls delta chunks.
 * Run: npx tsx examples/tool-calling-stream.ts
 */
import { LocalLLM } from 'local-llm';
import type { ChatCompletionTool } from 'local-llm';

const tools: ChatCompletionTool[] = [
  {
    type: 'function',
    function: {
      name: 'calculate',
      description: 'Evaluate a math expression',
      parameters: {
        type: 'object',
        properties: {
          expression: { type: 'string', description: 'Math expression, e.g. "2 + 2"' },
        },
        required: ['expression'],
      },
    },
  },
];

const ai = await LocalLLM.create({
  model: 'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
});

const stream = await ai.chat.completions.create({
  messages: [{ role: 'user', content: 'What is 42 * 17?' }],
  tools,
  tool_choice: 'required',
  stream: true,
});

let toolName = '';
let toolArgs = '';

for await (const chunk of stream) {
  const delta = chunk.choices[0].delta;

  if (delta.tool_calls) {
    for (const tc of delta.tool_calls) {
      if (tc.function?.name) {
        toolName = tc.function.name;
        console.log(`Tool: ${toolName}`);
      }
      if (tc.function?.arguments) {
        toolArgs += tc.function.arguments;
      }
    }
  }

  if (delta.content) {
    process.stdout.write(delta.content);
  }

  if (chunk.choices[0].finish_reason === 'tool_calls') {
    console.log(`Arguments: ${toolArgs}`);
    console.log('Stream finished with tool call');
  }
}

ai.dispose();
