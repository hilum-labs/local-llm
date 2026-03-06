import NativeLocalLLM from './NativeLocalLLM';
import { NativeEventEmitter, NativeModules } from 'react-native';
import type { NativeAddon, NativeModel, NativeContext, NativeMtmdContext } from 'local-llm/native';

const emitter = new NativeEventEmitter(NativeModules.LocalLLM);

type BrandedId<B extends string> = { __id: string; __brand: B };
const wrapId = <B extends string>(id: string, brand: B) => ({ __id: id, __brand: brand } as unknown);
const unwrapId = (obj: any): string => obj.__id;

export function createReactNativeAddon(): NativeAddon {
  return {
    backendInfo: () => NativeLocalLLM.backendInfo(),
    backendVersion: () => NativeLocalLLM.backendVersion(),

    loadModel: (path, options) => {
      const promise = NativeLocalLLM.loadModel(path, options ?? {});
      return promise.then((id) => wrapId(id, 'NativeModel')) as any;
    },
    loadModelFromBuffer: () => { throw new Error('loadModelFromBuffer is not available on React Native'); },
    getModelSize: (model) => NativeLocalLLM.getModelSize(unwrapId(model)),
    freeModel: (model) => NativeLocalLLM.freeModel(unwrapId(model)),

    createContext: (model, options) => {
      const id = NativeLocalLLM.createContext(unwrapId(model), options ?? {});
      return wrapId(id, 'NativeContext') as NativeContext;
    },
    getContextSize: (ctx) => NativeLocalLLM.getContextSize(unwrapId(ctx)),
    freeContext: (ctx) => NativeLocalLLM.freeContext(unwrapId(ctx)),

    kvCacheClear: (ctx, fromPos) => NativeLocalLLM.kvCacheClear(unwrapId(ctx), fromPos),

    tokenize: (model, text, addSpecial, parseSpecial) => {
      const arr = NativeLocalLLM.tokenize(unwrapId(model), text, addSpecial ?? true, parseSpecial ?? false);
      return new Int32Array(arr);
    },
    detokenize: (model, tokens) => {
      const plain = tokens instanceof Int32Array ? Array.from(tokens) : tokens;
      return NativeLocalLLM.detokenize(unwrapId(model), plain);
    },
    applyChatTemplate: (model, messages, addAssistant) =>
      NativeLocalLLM.applyChatTemplate(unwrapId(model), messages, addAssistant ?? true),

    inferSync: (model, ctx, prompt, options) =>
      NativeLocalLLM.generate(unwrapId(model), unwrapId(ctx), prompt, options ?? {}) as any,

    inferStream: (model, ctx, prompt, options, callback) => {
      const ctxId = unwrapId(ctx);
      const sub = emitter.addListener('onToken', (event) => {
        if (event.contextId !== ctxId) return;
        if (event.error) { callback(new Error(event.error), null); sub.remove(); return; }
        if (event.done) { callback(null, null); sub.remove(); return; }
        callback(null, event.token);
      });
      NativeLocalLLM.startStream(unwrapId(model), ctxId, prompt, options ?? {});
    },

    createMtmdContext: (model, projectorPath, options) => {
      const id = NativeLocalLLM.loadProjector(unwrapId(model), projectorPath, options ?? {});
      return wrapId(id, 'NativeMtmdContext') as NativeMtmdContext;
    },
    supportVision: (ctx) => NativeLocalLLM.supportVision(unwrapId(ctx)),
    freeMtmdContext: (ctx) => NativeLocalLLM.freeMtmdContext(unwrapId(ctx)),

    inferSyncVision: async (model, ctx, mtmdCtx, prompt, imageBuffers, options) => {
      const base64s = imageBuffers.map((buf) => buf.toString('base64'));
      return NativeLocalLLM.generateVision(
        unwrapId(model), unwrapId(ctx), unwrapId(mtmdCtx), prompt, base64s, options ?? {},
      );
    },

    inferStreamVision: (model, ctx, mtmdCtx, prompt, imageBuffers, options, callback) => {
      const ctxId = unwrapId(ctx);
      const base64s = imageBuffers.map((buf) => buf.toString('base64'));
      const sub = emitter.addListener('onToken', (event) => {
        if (event.contextId !== ctxId) return;
        if (event.error) { callback(new Error(event.error), null); sub.remove(); return; }
        if (event.done) { callback(null, null); sub.remove(); return; }
        callback(null, event.token);
      });
      NativeLocalLLM.startStreamVision(
        unwrapId(model), ctxId, unwrapId(mtmdCtx), prompt, base64s, options ?? {},
      );
    },

    jsonSchemaToGrammar: (schemaJson) => NativeLocalLLM.jsonSchemaToGrammar(schemaJson),

    getEmbeddingDimension: (model) => NativeLocalLLM.getEmbeddingDimension(unwrapId(model)),
    createEmbeddingContext: (model, options) => {
      const id = NativeLocalLLM.createEmbeddingContext(unwrapId(model), options ?? {});
      return wrapId(id, 'NativeContext') as NativeContext;
    },
    embed: (ctx, model, tokens) => {
      const plain = tokens instanceof Int32Array ? Array.from(tokens) : tokens;
      return new Float32Array(NativeLocalLLM.embed(unwrapId(ctx), unwrapId(model), plain));
    },
    embedBatch: (ctx, model, tokenArrays) => {
      const plainArrays = tokenArrays.map((t) => (t instanceof Int32Array ? Array.from(t) : t));
      const results = NativeLocalLLM.embedBatch(unwrapId(ctx), unwrapId(model), plainArrays) as number[][];
      return results.map((r) => new Float32Array(r));
    },

    inferBatch: (model, ctx, prompts, options, callback) => {
      const ctxId = unwrapId(ctx);
      let completedCount = 0;
      const sub = emitter.addListener('onBatchToken', (event) => {
        if (event.contextId !== ctxId) return;
        if (event.error) { callback(new Error(event.error), null, event.seqIndex, null); return; }
        if (event.done) {
          callback(null, null, event.seqIndex, event.finishReason ?? null);
          completedCount++;
          if (completedCount >= prompts.length) sub.remove();
          return;
        }
        callback(null, event.token, event.seqIndex, null);
      });
      NativeLocalLLM.startBatch(unwrapId(model), ctxId, prompts, options as any);
    },

    quantize: (inputPath, outputPath, options, callback) => {
      const sub = emitter.addListener('onQuantizeComplete', (event) => {
        sub.remove();
        callback(event.error ? new Error(event.error) : null);
      });
      NativeLocalLLM.quantize(inputPath, outputPath, options);
    },

    setLogCallback: (() => {
      let logSub: { remove(): void } | null = null;
      return (cb: ((level: number, text: string) => void) | null) => {
        if (logSub) { logSub.remove(); logSub = null; }
        if (cb) {
          logSub = emitter.addListener('onLog', (event) => cb(event.level, event.text));
          NativeLocalLLM.enableLogEvents(true);
        } else {
          NativeLocalLLM.enableLogEvents(false);
        }
      };
    })(),
    setLogLevel: (level) => NativeLocalLLM.setLogLevel(level),
  };
}
