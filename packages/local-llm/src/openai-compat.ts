import {
  ChatCompletions as CoreChatCompletions,
  type ChatCompletionsConfig,
  type VisionPromptBuilder,
} from 'local-llm-js-core';
import { buildVisionPrompt } from './engine.js';
import type { InferenceContext, Model } from './engine.js';

export type { ChatCompletionsConfig } from 'local-llm-js-core';

export class ChatCompletions extends CoreChatCompletions {
  constructor(
    model: Model,
    context: InferenceContext,
    modelName: string,
    config?: ChatCompletionsConfig,
  ) {
    const visionPromptBuilder = (config?.visionPromptBuilder
      ?? (async (messages) => buildVisionPrompt(messages))) as VisionPromptBuilder;
    super(model, context, modelName, { ...config, visionPromptBuilder });
  }
}
