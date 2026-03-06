import { describe, it, expect, vi, beforeEach } from 'vitest';

const mockNative = {
  backendInfo: vi.fn(() => 'Metal'),
  backendVersion: vi.fn(() => '1.0.0'),
  loadModel: vi.fn(() => ({ __brand: 'NativeModel' })),
  loadModelFromBuffer: vi.fn(() => ({ __brand: 'NativeModel' })),
  createContext: vi.fn(() => ({ __brand: 'NativeContext' })),
  tokenize: vi.fn((_m: any, text: string) => {
    // Return deterministic token IDs based on char codes for prefix matching tests
    const arr = new Int32Array(text.length);
    for (let i = 0; i < text.length; i++) arr[i] = text.charCodeAt(i);
    return arr;
  }),
  detokenize: vi.fn(() => 'text'),
  applyChatTemplate: vi.fn((_m: any, msgs: any[]) => 'prompt'),
  inferSync: vi.fn(() => 'Hello, world!'),
  inferSyncVision: vi.fn(() => 'A cat'),
  inferStream: vi.fn((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
    cb(null, 'Hello');
    cb(null, ', ');
    cb(null, 'world!');
    cb(null, null);
  }),
  inferStreamVision: vi.fn((_m: any, _c: any, _mtmd: any, _p: string, _imgs: any, _o: any, cb: Function) => {
    cb(null, 'A ');
    cb(null, 'cat');
    cb(null, null);
  }),
  getModelSize: vi.fn(() => 1000),
  freeModel: vi.fn(),
  freeContext: vi.fn(),
  kvCacheClear: vi.fn(),
  createMtmdContext: vi.fn(() => ({ __brand: 'NativeMtmdContext' })),
  freeMtmdContext: vi.fn(),
  supportVision: vi.fn(() => true),
  setLogCallback: vi.fn(),
  setLogLevel: vi.fn(),
};

vi.mock('./native.js', () => ({
  loadNativeAddon: () => mockNative,
}));

const { Model, InferenceContext } = await import('./engine.js');

beforeEach(() => {
  vi.clearAllMocks();
  mockNative.loadModel.mockReturnValue({ __brand: 'NativeModel' });
  mockNative.loadModelFromBuffer.mockReturnValue({ __brand: 'NativeModel' });
  mockNative.createContext.mockReturnValue({ __brand: 'NativeContext' });
  mockNative.createMtmdContext.mockReturnValue({ __brand: 'NativeMtmdContext' });
  mockNative.inferSync.mockReturnValue('Hello, world!');
  mockNative.inferSyncVision.mockReturnValue('A cat');
  mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
    cb(null, 'Hello');
    cb(null, ', ');
    cb(null, 'world!');
    cb(null, null);
  });
  mockNative.inferStreamVision.mockImplementation((_m: any, _c: any, _mtmd: any, _p: string, _imgs: any, _o: any, cb: Function) => {
    cb(null, 'A ');
    cb(null, 'cat');
    cb(null, null);
  });
});

describe('Model', () => {
  describe('constructor error wrapping', () => {
    it('wraps file-not-found errors with helpful message', () => {
      mockNative.loadModel.mockImplementation(() => {
        throw new Error('no such file or directory');
      });
      expect(() => new Model('/bad/path.gguf')).toThrow('Model file not found');
    });

    it('wraps memory errors with helpful message', () => {
      mockNative.loadModel.mockImplementation(() => {
        throw new Error('failed to alloc memory');
      });
      expect(() => new Model('/model.gguf')).toThrow('insufficient memory');
    });

    it('wraps unknown errors with generic message', () => {
      mockNative.loadModel.mockImplementation(() => {
        throw new Error('something weird');
      });
      expect(() => new Model('/model.gguf')).toThrow('Failed to load model');
    });
  });

  describe('fromBuffer error wrapping', () => {
    it('wraps memory errors from buffer loading', () => {
      mockNative.loadModelFromBuffer.mockImplementation(() => {
        throw new Error('mmap failed');
      });
      expect(() => Model.fromBuffer(Buffer.from('test'))).toThrow('insufficient memory');
    });
  });

  describe('createContext error wrapping', () => {
    it('wraps memory errors with helpful message', () => {
      const model = new Model('/model.gguf');
      mockNative.createContext.mockImplementation(() => {
        throw new Error('failed to alloc');
      });
      expect(() => model.createContext()).toThrow('insufficient memory');
      expect(() => model.createContext()).toThrow('contextSize');
    });
  });

  describe('loadProjector error wrapping', () => {
    it('wraps file-not-found errors', () => {
      const model = new Model('/model.gguf');
      mockNative.createMtmdContext.mockImplementation(() => {
        throw new Error('file not found');
      });
      expect(() => model.loadProjector('/bad/proj.gguf')).toThrow('Projector file not found');
    });
  });
});

describe('InferenceContext', () => {
  function createContext() {
    const model = new Model('/model.gguf');
    return model.createContext();
  }

  describe('generate', () => {
    it('returns generated text', async () => {
      const ctx = createContext();
      const text = await ctx.generate('prompt');
      expect(text).toBe('Hello, world!');
    });

    it('uses inferStream internally (non-blocking)', async () => {
      const ctx = createContext();
      await ctx.generate('prompt');
      expect(mockNative.inferStream).toHaveBeenCalled();
      expect(mockNative.inferSync).not.toHaveBeenCalled();
    });

    it('truncates at stop sequence', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'Hello, world! END more text');
        cb(null, null);
      });
      const ctx = createContext();
      const text = await ctx.generate('prompt', { stop: ['END'] });
      expect(text).toBe('Hello, world! ');
    });

    it('handles multiple stop sequences (picks earliest)', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'Hello\n\nworld\n---\nmore');
        cb(null, null);
      });
      const ctx = createContext();
      const text = await ctx.generate('prompt', { stop: ['\n---', '\n\n'] });
      expect(text).toBe('Hello');
    });

    it('returns full text when no stop sequence is found', async () => {
      const ctx = createContext();
      const text = await ctx.generate('prompt', { stop: ['NOTFOUND'] });
      expect(text).toBe('Hello, world!');
    });

    it('rejects when signal is already aborted', async () => {
      const ctx = createContext();
      const controller = new AbortController();
      controller.abort();
      await expect(ctx.generate('prompt', { signal: controller.signal })).rejects.toThrow('aborted');
    });

    it('wraps native inference errors', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(new Error('segfault'), null);
      });
      const ctx = createContext();
      await expect(ctx.generate('prompt')).rejects.toThrow('Inference failed');
    });
  });

  describe('stream', () => {
    it('yields tokens', async () => {
      const ctx = createContext();
      const tokens: string[] = [];
      for await (const token of ctx.stream('prompt')) {
        tokens.push(token);
      }
      expect(tokens).toEqual(['Hello', ', ', 'world!']);
    });

    it('handles stop sequences across token boundaries', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'He');
        cb(null, 'llo');
        cb(null, ' EN');
        cb(null, 'D more');
        cb(null, null);
      });
      const ctx = createContext();
      const tokens: string[] = [];
      for await (const token of ctx.stream('prompt', { stop: ['END'] })) {
        tokens.push(token);
      }
      expect(tokens.join('')).toBe('Hello ');
    });

    it('handles stop sequence at the very start', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'STOP');
        cb(null, ' more');
        cb(null, null);
      });
      const ctx = createContext();
      const tokens: string[] = [];
      for await (const token of ctx.stream('prompt', { stop: ['STOP'] })) {
        tokens.push(token);
      }
      expect(tokens.join('')).toBe('');
    });

    it('yields all text when stop sequence not found', async () => {
      const ctx = createContext();
      const tokens: string[] = [];
      for await (const token of ctx.stream('prompt', { stop: ['NOTFOUND'] })) {
        tokens.push(token);
      }
      expect(tokens.join('')).toBe('Hello, world!');
    });

    it('stops on abort signal', async () => {
      const controller = new AbortController();
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'token1');
        setTimeout(() => {
          controller.abort();
          cb(null, 'token2');
          cb(null, null);
        }, 10);
      });
      const ctx = createContext();
      const tokens: string[] = [];
      for await (const token of ctx.stream('prompt', { signal: controller.signal })) {
        tokens.push(token);
      }
      expect(tokens.length).toBeGreaterThanOrEqual(1);
    });

    it('throws when signal is already aborted', async () => {
      const controller = new AbortController();
      controller.abort();
      const ctx = createContext();
      const gen = ctx.stream('prompt', { signal: controller.signal });
      await expect(gen.next()).rejects.toThrow('aborted');
    });

    it('propagates native errors from stream', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(new Error('stream error'), null);
      });
      const ctx = createContext();
      const gen = ctx.stream('prompt');
      await expect(gen.next()).rejects.toThrow('stream error');
    });

    it('flushes buffer at end when using stop sequences with no match', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'Hi');
        cb(null, null);
      });
      const ctx = createContext();
      const tokens: string[] = [];
      for await (const token of ctx.stream('prompt', { stop: ['NOPE'] })) {
        tokens.push(token);
      }
      expect(tokens.join('')).toBe('Hi');
    });
  });

  describe('generateVision', () => {
    it('uses inferStreamVision internally (non-blocking)', async () => {
      const model = new Model('/model.gguf');
      model.loadProjector('/proj.gguf');
      const ctx = model.createContext();
      await ctx.generateVision('prompt', [Buffer.from('img')]);
      expect(mockNative.inferStreamVision).toHaveBeenCalled();
      expect(mockNative.inferSyncVision).not.toHaveBeenCalled();
    });

    it('truncates at stop sequence', async () => {
      mockNative.inferStreamVision.mockImplementation((_m: any, _c: any, _mtmd: any, _p: string, _imgs: any, _o: any, cb: Function) => {
        cb(null, 'A cat END extra');
        cb(null, null);
      });
      const model = new Model('/model.gguf');
      model.loadProjector('/proj.gguf');
      const ctx = model.createContext();
      const text = await ctx.generateVision('prompt', [Buffer.from('img')], { stop: ['END'] });
      expect(text).toBe('A cat ');
    });

    it('rejects when signal is already aborted', async () => {
      const model = new Model('/model.gguf');
      model.loadProjector('/proj.gguf');
      const ctx = model.createContext();
      const controller = new AbortController();
      controller.abort();
      await expect(
        ctx.generateVision('prompt', [Buffer.from('img')], { signal: controller.signal }),
      ).rejects.toThrow('aborted');
    });
  });

  describe('streamVision', () => {
    function createVisionContext() {
      const model = new Model('/model.gguf');
      model.loadProjector('/proj.gguf');
      return model.createContext();
    }

    it('yields tokens', async () => {
      const ctx = createVisionContext();
      const tokens: string[] = [];
      for await (const token of ctx.streamVision('prompt', [Buffer.from('img')])) {
        tokens.push(token);
      }
      expect(tokens).toEqual(['A ', 'cat']);
    });

    it('handles stop sequences', async () => {
      mockNative.inferStreamVision.mockImplementation((_m: any, _c: any, _mtmd: any, _p: string, _imgs: any, _o: any, cb: Function) => {
        cb(null, 'A cat');
        cb(null, ' END');
        cb(null, ' more');
        cb(null, null);
      });
      const ctx = createVisionContext();
      const tokens: string[] = [];
      for await (const token of ctx.streamVision('prompt', [Buffer.from('img')], { stop: ['END'] })) {
        tokens.push(token);
      }
      expect(tokens.join('')).toBe('A cat ');
    });

    it('stops on abort signal', async () => {
      const controller = new AbortController();
      mockNative.inferStreamVision.mockImplementation((_m: any, _c: any, _mtmd: any, _p: string, _imgs: any, _o: any, cb: Function) => {
        cb(null, 'token1');
        setTimeout(() => {
          controller.abort();
          cb(null, 'token2');
          cb(null, null);
        }, 10);
      });
      const ctx = createVisionContext();
      const tokens: string[] = [];
      for await (const token of ctx.streamVision('prompt', [Buffer.from('img')], { signal: controller.signal })) {
        tokens.push(token);
      }
      expect(tokens.length).toBeGreaterThanOrEqual(1);
    });

    it('throws when signal is already aborted', async () => {
      const controller = new AbortController();
      controller.abort();
      const ctx = createVisionContext();
      const gen = ctx.streamVision('prompt', [Buffer.from('img')], { signal: controller.signal });
      await expect(gen.next()).rejects.toThrow('aborted');
    });

    it('throws when no projector loaded', async () => {
      const ctx = createContext();
      const gen = ctx.streamVision('prompt', [Buffer.from('img')]);
      await expect(gen.next()).rejects.toThrow('No projector loaded');
    });

    it('propagates native errors', async () => {
      mockNative.inferStreamVision.mockImplementation((_m: any, _c: any, _mtmd: any, _p: string, _imgs: any, _o: any, cb: Function) => {
        cb(new Error('native error'), null);
      });
      const ctx = createVisionContext();
      const gen = ctx.streamVision('prompt', [Buffer.from('img')]);
      await expect(gen.next()).rejects.toThrow('native error');
    });
  });

  describe('concurrency', () => {
    it('serializes concurrent generate() calls', async () => {
      let activeCount = 0;
      let maxConcurrent = 0;

      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        activeCount++;
        maxConcurrent = Math.max(maxConcurrent, activeCount);
        setTimeout(() => {
          cb(null, 'result');
          cb(null, null);
          activeCount--;
        }, 10);
      });

      const ctx = createContext();
      const [r1, r2] = await Promise.all([
        ctx.generate('prompt1'),
        ctx.generate('prompt2'),
      ]);

      expect(r1).toBe('result');
      expect(r2).toBe('result');
      expect(maxConcurrent).toBe(1);
    });

    it('serializes concurrent stream() calls', async () => {
      let activeStreams = 0;
      let maxConcurrent = 0;

      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        activeStreams++;
        maxConcurrent = Math.max(maxConcurrent, activeStreams);
        setTimeout(() => {
          cb(null, 'token');
          cb(null, null);
          activeStreams--;
        }, 10);
      });

      const ctx = createContext();

      const consume = async (gen: AsyncGenerator<string>) => {
        const tokens: string[] = [];
        for await (const t of gen) tokens.push(t);
        return tokens;
      };

      await Promise.all([
        consume(ctx.stream('p1')),
        consume(ctx.stream('p2')),
      ]);

      expect(maxConcurrent).toBe(1);
    });

    it('serializes generate() and stream() calls against each other', async () => {
      let activeCount = 0;
      let maxConcurrent = 0;
      const callOrder: string[] = [];

      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        activeCount++;
        maxConcurrent = Math.max(maxConcurrent, activeCount);
        callOrder.push('start');
        setTimeout(() => {
          cb(null, 'tok');
          cb(null, null);
          callOrder.push('end');
          activeCount--;
        }, 10);
      });

      const ctx = createContext();

      const streamPromise = (async () => {
        const tokens: string[] = [];
        for await (const t of ctx.stream('p1')) tokens.push(t);
        return tokens;
      })();

      const genPromise = ctx.generate('p2');

      await Promise.all([streamPromise, genPromise]);

      expect(maxConcurrent).toBe(1);
    });

    it('releases mutex even when generate() throws', async () => {
      mockNative.inferStream
        .mockImplementationOnce((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
          cb(new Error('fail'), null);
        })
        .mockImplementationOnce((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
          cb(null, 'ok');
          cb(null, null);
        });

      const ctx = createContext();

      await expect(ctx.generate('p1')).rejects.toThrow('fail');
      const result = await ctx.generate('p2');
      expect(result).toBe('ok');
    });

    it('releases mutex even when stream() throws', async () => {
      mockNative.inferStream
        .mockImplementationOnce((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
          cb(new Error('stream fail'), null);
        })
        .mockImplementationOnce((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
          cb(null, 'ok');
          cb(null, null);
        });

      const ctx = createContext();

      const gen1 = ctx.stream('p1');
      await expect(gen1.next()).rejects.toThrow('stream fail');

      const tokens: string[] = [];
      for await (const t of ctx.stream('p2')) tokens.push(t);
      expect(tokens).toEqual(['ok']);
    });
  });

  describe('KV cache reuse', () => {
    it('passes n_past=0 on first call', async () => {
      const ctx = createContext();
      await ctx.generate('hello');
      expect(mockNative.inferStream).toHaveBeenCalledWith(
        expect.anything(), expect.anything(), 'hello',
        expect.objectContaining({ n_past: 0 }),
        expect.any(Function),
      );
    });

    it('passes n_past > 0 when prompt shares prefix with previous call', async () => {
      // First call: prompt "abc" — generates "xyz"
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'xyz');
        cb(null, null);
      });
      const ctx = createContext();
      await ctx.generate('abc');

      // Second call: prompt "abcxyz_more" — shares prefix "abc" + "xyz" (6 chars)
      // with the cached tokens [a,b,c,x,y,z]
      await ctx.generate('abcxyz_more');

      const secondCallOpts = mockNative.inferStream.mock.calls[1][3];
      expect(secondCallOpts.n_past).toBe(6); // "abcxyz" is the common prefix
    });

    it('clears KV cache when prompt diverges from cached tokens', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'result');
        cb(null, null);
      });
      const ctx = createContext();

      // First call: "hello"
      await ctx.generate('hello');
      expect(mockNative.kvCacheClear).not.toHaveBeenCalled();

      // Second call: "world" — completely different, no common prefix
      await ctx.generate('world');
      // Should clear from position 0 (no common prefix)
      expect(mockNative.kvCacheClear).toHaveBeenCalledWith(expect.anything(), 0);
    });

    it('does not clear KV cache when prompt extends previous', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'x');
        cb(null, null);
      });
      const ctx = createContext();

      // First call: "ab" → generates "x", cache = [a,b,x]
      await ctx.generate('ab');
      mockNative.kvCacheClear.mockClear();

      // Second call: "abx" — exactly matches cached [a,b,x], so n_past=3
      await ctx.generate('abx');
      expect(mockNative.kvCacheClear).not.toHaveBeenCalled();
      const opts = mockNative.inferStream.mock.calls[1][3];
      expect(opts.n_past).toBe(3);
    });

    it('works with streaming too', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'x');
        cb(null, null);
      });
      const ctx = createContext();

      // Stream first call
      const tokens1: string[] = [];
      for await (const t of ctx.stream('ab')) tokens1.push(t);

      // Stream second call with extended prompt
      const tokens2: string[] = [];
      for await (const t of ctx.stream('abx_more')) tokens2.push(t);

      const opts = mockNative.inferStream.mock.calls[1][3];
      expect(opts.n_past).toBe(3); // "abx" is common prefix
    });

    it('invalidates cache on vision calls', async () => {
      mockNative.inferStream.mockImplementation((_m: any, _c: any, _p: string, _o: any, cb: Function) => {
        cb(null, 'text');
        cb(null, null);
      });
      const model = new Model('/model.gguf');
      model.loadProjector('/proj.gguf');
      const ctx = model.createContext();

      // Text call — builds cache
      await ctx.generate('hello');

      // Vision call — should invalidate cache
      await ctx.generateVision('describe', [Buffer.from('img')]);

      // Text call again — should start fresh (n_past=0)
      await ctx.generate('hello');
      const lastCallOpts = mockNative.inferStream.mock.calls[mockNative.inferStream.mock.calls.length - 1][3];
      expect(lastCallOpts.n_past).toBe(0);
    });
  });

  describe('dispose', () => {
    it('prevents further generation', async () => {
      const ctx = createContext();
      ctx.dispose();
      await expect(ctx.generate('prompt')).rejects.toThrow('Context has been disposed');
    });
  });
});
