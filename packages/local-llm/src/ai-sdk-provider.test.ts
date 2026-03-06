import { describe, it, expect, vi } from 'vitest';
import { LocalAILanguageModel } from './ai-sdk-provider.js';

function createMockModel(options?: { tokenizeLength?: number }) {
  const tokenLen = options?.tokenizeLength ?? 5;
  return {
    tokenize: vi.fn((_text: string) => new Int32Array(tokenLen)),
    applyChatTemplate: vi.fn(
      (_messages: unknown[], _addAssistant?: boolean) => 'formatted-prompt',
    ),
    detokenize: vi.fn(),
    createContext: vi.fn(),
    dispose: vi.fn(),
    sizeBytes: 1000,
  };
}

function createMockContext(options?: {
  generateResult?: string;
  streamTokens?: string[];
}) {
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

describe('LocalAILanguageModel', () => {
  describe('properties', () => {
    it('has correct specificationVersion, provider, modelId', () => {
      const model = createMockModel();
      const context = createMockContext();
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      expect(lm.specificationVersion).toBe('v3');
      expect(lm.provider).toBe('local-llm');
      expect(lm.modelId).toBe('test-model');
      expect(lm.supportedUrls).toEqual({});
    });
  });

  describe('doGenerate', () => {
    it('returns correct result shape', async () => {
      const model = createMockModel();
      const context = createMockContext({ generateResult: 'Generated text' });
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const result = await lm.doGenerate({
        prompt: [{ role: 'user', content: 'Hello' }],
      });

      expect(result.content).toEqual([{ type: 'text', text: 'Generated text' }]);
      expect(result.finishReason).toBe('stop');
      expect(result.usage).toEqual({ inputTokens: 5, outputTokens: 5 });
      expect(result.warnings).toEqual([]);
    });

    it('finishReason is "length" when output reaches maxOutputTokens', async () => {
      const model = createMockModel({ tokenizeLength: 10 });
      const context = createMockContext({ generateResult: 'Long text' });
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const result = await lm.doGenerate({
        prompt: [{ role: 'user', content: 'Hello' }],
        maxOutputTokens: 10,
      });

      expect(result.finishReason).toBe('length');
    });

    it('finishReason is "stop" when output is shorter than maxOutputTokens', async () => {
      const model = createMockModel({ tokenizeLength: 3 });
      const context = createMockContext({ generateResult: 'Short' });
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const result = await lm.doGenerate({
        prompt: [{ role: 'user', content: 'Hello' }],
        maxOutputTokens: 100,
      });

      expect(result.finishReason).toBe('stop');
    });
  });

  describe('prompt conversion', () => {
    it('converts system/user/assistant messages', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      await lm.doGenerate({
        prompt: [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there' },
        ],
      });

      expect(model.applyChatTemplate).toHaveBeenCalledWith(
        [
          { role: 'system', content: 'You are helpful.' },
          { role: 'user', content: 'Hello' },
          { role: 'assistant', content: 'Hi there' },
        ],
        true,
      );
    });

    it('converts tool messages to user messages', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      await lm.doGenerate({
        prompt: [
          { role: 'user', content: 'Call the tool' },
          { role: 'tool', content: 'tool result data' },
        ],
      });

      expect(model.applyChatTemplate).toHaveBeenCalledWith(
        [
          { role: 'user', content: 'Call the tool' },
          { role: 'user', content: 'tool result data' },
        ],
        true,
      );
    });

    it('handles array content parts (text type)', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      await lm.doGenerate({
        prompt: [
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Hello ' },
              { type: 'text', text: 'world' },
            ],
          },
        ],
      });

      expect(model.applyChatTemplate).toHaveBeenCalledWith(
        [{ role: 'user', content: 'Hello world' }],
        true,
      );
    });
  });

  describe('warnings', () => {
    it('handles JSON response format without warning', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const result = await lm.doGenerate({
        prompt: [{ role: 'user', content: 'Hello' }],
        responseFormat: { type: 'json' },
      });

      expect(result.warnings).toEqual([]);
    });

    it('warns about tools', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const result = await lm.doGenerate({
        prompt: [{ role: 'user', content: 'Hello' }],
        tools: [{ name: 'test_tool' }],
      });

      expect(result.warnings).toContainEqual(
        expect.objectContaining({
          type: 'unsupported-setting',
          message: expect.stringContaining('Tool use'),
        }),
      );
    });

    it('passes presencePenalty without warning', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const result = await lm.doGenerate({
        prompt: [{ role: 'user', content: 'Hello' }],
        presencePenalty: 0.5,
      });

      expect(result.warnings).toEqual([]);
    });

    it('no warnings when no unsupported settings', async () => {
      const model = createMockModel();
      const context = createMockContext();
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const result = await lm.doGenerate({
        prompt: [{ role: 'user', content: 'Hello' }],
      });

      expect(result.warnings).toEqual([]);
    });
  });

  describe('image content', () => {
    it('routes to generateVision when image parts are present', async () => {
      const model = createMockModel();
      const context = createMockContext({ generateResult: 'A cat' });
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const result = await lm.doGenerate({
        prompt: [{
          role: 'user',
          content: [
            { type: 'text', text: 'What is this?' },
            { type: 'image', image: new Uint8Array([0x89, 0x50, 0x4e, 0x47]), mimeType: 'image/png' },
          ],
        }],
      });

      expect(result.content[0].text).toBe('A cat');
      expect(context.generateVision).toHaveBeenCalled();
      expect(context.generate).not.toHaveBeenCalled();
    });

    it('uses text-only path when no image parts', async () => {
      const model = createMockModel();
      const context = createMockContext({ generateResult: 'Hello' });
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      await lm.doGenerate({
        prompt: [{ role: 'user', content: [{ type: 'text', text: 'Hello' }] }],
      });

      expect(context.generate).toHaveBeenCalled();
      expect(context.generateVision).not.toHaveBeenCalled();
    });
  });

  describe('doStream', () => {
    it('emits stream-start, text-start, text-delta(s), text-end, finish', async () => {
      const model = createMockModel({ tokenizeLength: 2 });
      const context = createMockContext({ streamTokens: ['Hi', ' there'] });
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const { stream } = await lm.doStream({
        prompt: [{ role: 'user', content: 'Hello' }],
      });

      const parts: Array<{ type: string; [key: string]: unknown }> = [];
      const reader = stream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
      }

      expect(parts[0].type).toBe('stream-start');
      expect(parts[1].type).toBe('text-start');
      expect(parts[2]).toEqual({ type: 'text-delta', textDelta: 'Hi' });
      expect(parts[3]).toEqual({ type: 'text-delta', textDelta: ' there' });
      expect(parts[4].type).toBe('text-end');
      expect(parts[5].type).toBe('finish');
      expect(parts[5].finishReason).toBe('stop');
      expect(parts[5].usage).toEqual({ inputTokens: 2, outputTokens: 2 });
    });

    it('includes warnings in stream-start for unsupported settings', async () => {
      const model = createMockModel();
      const context = createMockContext({ streamTokens: ['A'] });
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const { stream } = await lm.doStream({
        prompt: [{ role: 'user', content: 'Hello' }],
        tools: [{ type: 'function', function: { name: 'test' } }],
      });

      const reader = stream.getReader();
      const { value: first } = await reader.read();
      expect(first!.type).toBe('stream-start');
      expect((first as any).warnings).toHaveLength(1);
      reader.releaseLock();
    });

    it('finishReason is "length" when tokens reach maxOutputTokens', async () => {
      const model = createMockModel({ tokenizeLength: 10 });
      const context = createMockContext({ streamTokens: ['tokens'] });
      const lm = new LocalAILanguageModel(model as any, context as any, 'test-model');

      const { stream } = await lm.doStream({
        prompt: [{ role: 'user', content: 'Hello' }],
        maxOutputTokens: 10,
      });

      const parts: Array<{ type: string; [key: string]: unknown }> = [];
      const reader = stream.getReader();
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        parts.push(value);
      }

      const finish = parts.find((p) => p.type === 'finish');
      expect(finish!.finishReason).toBe('length');
    });
  });
});
