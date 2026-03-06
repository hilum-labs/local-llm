import { describe, it, expect, vi } from 'vitest';
import { ChatCompletions } from './openai-compat.js';

// Mock Model and InferenceContext to avoid loading native addons

function createMockModel(options?: { tokenizeLength?: number }) {
  const tokenLen = options?.tokenizeLength ?? 5;
  return {
    tokenize: vi.fn((text: string) => new Int32Array(tokenLen)),
    applyChatTemplate: vi.fn((messages: unknown[], addAssistant?: boolean) => 'formatted-prompt'),
    detokenize: vi.fn(),
    createContext: vi.fn(),
    dispose: vi.fn(),
    sizeBytes: 1000,
  };
}

function createMockContext(options?: { generateResult?: string; streamTokens?: string[] }) {
  const generateResult = options?.generateResult ?? 'Hello, world!';
  const streamTokens = options?.streamTokens ?? ['Hello', ',', ' world', '!'];

  return {
    generate: vi.fn(async () => generateResult),
    generateVision: vi.fn(async () => generateResult),
    stream: vi.fn(async function* () {
      for (const token of streamTokens) {
        yield token;
      }
    }),
    streamVision: vi.fn(async function* () {
      for (const token of streamTokens) {
        yield token;
      }
    }),
    dispose: vi.fn(),
  };
}

describe('ChatCompletions', () => {
  describe('non-streaming', () => {
    it('returns correct response shape', async () => {
      const model = createMockModel({ tokenizeLength: 5 });
      const context = createMockContext({ generateResult: 'Test response' });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const response = await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
      });

      expect(response.id).toMatch(/^chatcmpl-/);
      expect(response.object).toBe('chat.completion');
      expect(response.created).toBeTypeOf('number');
      expect(response.model).toBe('test-model');
      expect(response.choices).toHaveLength(1);
      expect(response.choices[0].index).toBe(0);
      expect(response.choices[0].message.role).toBe('assistant');
      expect(response.choices[0].message.content).toBe('Test response');
      expect(response.usage).toBeDefined();
      expect(response.usage.prompt_tokens).toBeTypeOf('number');
      expect(response.usage.completion_tokens).toBeTypeOf('number');
      expect(response.usage.total_tokens).toBe(
        response.usage.prompt_tokens + response.usage.completion_tokens,
      );
    });

    it('uses the model name from params when provided', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const completions = new ChatCompletions(model as any, context as any, 'default-model');

      const response = await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        model: 'custom-model',
      });

      expect(response.model).toBe('custom-model');
    });

    it('finish_reason is "stop" when output is shorter than maxTokens', async () => {
      // Mock tokenize to return 3 tokens for the generated content
      const model = createMockModel({ tokenizeLength: 3 });
      const context = createMockContext({ generateResult: 'Short' });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const response = await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 100,
      });

      expect(response.choices[0].finish_reason).toBe('stop');
    });

    it('finish_reason is "length" when output reaches maxTokens', async () => {
      // Mock tokenize to return 10 tokens for the generated content
      const model = createMockModel({ tokenizeLength: 10 });
      const context = createMockContext({ generateResult: 'A long response here' });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const response = await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        max_tokens: 10,
      });

      expect(response.choices[0].finish_reason).toBe('length');
    });

    it('token usage counts are accurate', async () => {
      let callCount = 0;
      const model = createMockModel();
      // tokenize is called 3 times: once for context overflow check, once for prompt count, once for completion count
      // The first two calls tokenize the prompt (return 8), the third tokenizes the completion (return 5)
      model.tokenize = vi.fn(() => {
        callCount++;
        return new Int32Array(callCount <= 2 ? 8 : 5);
      });
      const context = createMockContext({ generateResult: 'Response' });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const response = await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
      });

      expect(response.usage.prompt_tokens).toBe(8);
      expect(response.usage.completion_tokens).toBe(5);
      expect(response.usage.total_tokens).toBe(13);
    });

    it('passes generation options through', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        temperature: 0.5,
        top_p: 0.8,
        top_k: 50,
        max_tokens: 200,
        seed: 42,
        frequency_penalty: 1.2,
      });

      expect(context.generate).toHaveBeenCalledWith('formatted-prompt', expect.objectContaining({
        maxTokens: 200,
        temperature: 0.5,
        topP: 0.8,
        topK: 50,
        seed: 42,
        frequencyPenalty: 1.2,
      }));
    });
  });

  describe('streaming', () => {
    it('yields role chunk, content chunks, and finish chunk in order', async () => {
      const model = createMockModel();
      const context = createMockContext({ streamTokens: ['Hi', ' there'] });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const stream = await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        stream: true,
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      // First chunk: role
      expect(chunks[0].choices[0].delta.role).toBe('assistant');
      expect(chunks[0].choices[0].finish_reason).toBeNull();

      // Content chunks
      expect(chunks[1].choices[0].delta.content).toBe('Hi');
      expect(chunks[1].choices[0].finish_reason).toBeNull();
      expect(chunks[2].choices[0].delta.content).toBe(' there');
      expect(chunks[2].choices[0].finish_reason).toBeNull();

      // Final chunk
      const last = chunks[chunks.length - 1];
      expect(last.choices[0].delta).toEqual({});
      expect(last.choices[0].finish_reason).toBe('stop');
    });

    it('all chunks share the same id, object, created, model', async () => {
      const model = createMockModel();
      const context = createMockContext({ streamTokens: ['A', 'B'] });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const stream = await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        stream: true,
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      const { id, object, created, model: m } = chunks[0];
      expect(id).toMatch(/^chatcmpl-/);
      expect(object).toBe('chat.completion.chunk');

      for (const chunk of chunks) {
        expect(chunk.id).toBe(id);
        expect(chunk.object).toBe(object);
        expect(chunk.created).toBe(created);
        expect(chunk.model).toBe(m);
      }
    });
  });

  describe('multimodal', () => {
    it('detects image content and calls generateVision for non-streaming', async () => {
      const model = createMockModel();
      const context = createMockContext({ generateResult: 'A cat' });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const imgBytes = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
      const dataUri = `data:image/png;base64,${imgBytes.toString('base64')}`;

      const response = await completions.create({
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            { type: 'image_url', image_url: { url: dataUri } },
          ],
        }],
      });

      expect(response.choices[0].message.content).toBe('A cat');
      expect(context.generateVision).toHaveBeenCalled();
      expect(context.generate).not.toHaveBeenCalled();
    });

    it('detects image content and calls streamVision for streaming', async () => {
      const model = createMockModel();
      const context = createMockContext({ streamTokens: ['A', ' cat'] });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const imgBytes = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
      const dataUri = `data:image/png;base64,${imgBytes.toString('base64')}`;

      const stream = await completions.create({
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            { type: 'image_url', image_url: { url: dataUri } },
          ],
        }],
        stream: true,
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(chunks.length).toBeGreaterThan(0);
      expect(context.streamVision).toHaveBeenCalled();
      expect(context.stream).not.toHaveBeenCalled();
    });

    it('uses text-only path when content is string', async () => {
      const model = createMockModel();
      const context = createMockContext({ generateResult: 'Hello' });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      await completions.create({
        messages: [{ role: 'user', content: 'Hi' }],
      });

      expect(context.generate).toHaveBeenCalled();
      expect(context.generateVision).not.toHaveBeenCalled();
    });

    it('uses text-only path when array content has no images', async () => {
      const model = createMockModel();
      const context = createMockContext({ generateResult: 'Hello' });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      await completions.create({
        messages: [{
          role: 'user',
          content: [
            { type: 'text', text: 'Hello' },
            { type: 'text', text: ' world' },
          ],
        }],
      });

      expect(context.generate).toHaveBeenCalled();
      expect(context.generateVision).not.toHaveBeenCalled();
    });
  });

  describe('stop sequences', () => {
    it('passes stop sequences through to generate options', async () => {
      const model = createMockModel();
      const context = createMockContext({ generateResult: 'Hello\n\nMore text' });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        stop: ['\n\n'],
      });

      expect(context.generate).toHaveBeenCalledWith(
        'formatted-prompt',
        expect.objectContaining({ stop: ['\n\n'] }),
      );
    });

    it('passes stop sequences for streaming', async () => {
      const model = createMockModel();
      const context = createMockContext({ streamTokens: ['Hi', ' there'] });
      const completions = new ChatCompletions(model as any, context as any, 'test-model');

      const stream = await completions.create({
        messages: [{ role: 'user', content: 'Hello' }],
        stream: true,
        stop: ['END'],
      });

      const chunks = [];
      for await (const chunk of stream) {
        chunks.push(chunk);
      }

      expect(context.stream).toHaveBeenCalledWith(
        'formatted-prompt',
        expect.objectContaining({ stop: ['END'] }),
      );
    });
  });
});
