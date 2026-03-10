import { LocalLLMProvider as CoreLocalLLMProvider, type LocalLLMProviderConfig } from 'local-llm-js-core';
import type { InferenceContext, Model } from './engine.js';

export class LocalLLMProvider extends CoreLocalLLMProvider {
  constructor(model: Model, context: InferenceContext, modelId: string, config?: LocalLLMProviderConfig) {
    super(model, context, modelId, {
      ...config,
      visionPromptBuilder: config?.visionPromptBuilder ?? (async (messages) => {
        const { buildVisionPrompt } = await import('./engine.js');
        return buildVisionPrompt(messages);
      }),
    });
  }
}
