import { readFile } from 'node:fs/promises';
import {
  EmbeddingContext,
  InferenceContext,
  Model,
  hasImageContent,
  optimalThreadCount,
  quantize,
  setNativeAddonLoader,
} from 'local-llm-js-core';
import { loadNativeAddon } from './native.js';
import type { ChatMessage, ContentPart } from './types.js';

setNativeAddonLoader(loadNativeAddon);

export { EmbeddingContext, InferenceContext, Model, hasImageContent, optimalThreadCount, quantize };

const MEDIA_MARKER = '<__media__>';

/** Resolve an image URL (data URI, file path, or HTTP URL) to a Buffer. */
export async function resolveImageToBuffer(url: string): Promise<Buffer> {
  if (url.startsWith('data:')) {
    const commaIdx = url.indexOf(',');
    if (commaIdx === -1) throw new Error('Invalid data URI');
    const base64 = url.slice(commaIdx + 1);
    return Buffer.from(base64, 'base64');
  }

  if (url.startsWith('http://') || url.startsWith('https://')) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to fetch image: ${res.status} ${res.statusText}`);
    return Buffer.from(await res.arrayBuffer());
  }

  return readFile(url);
}

/**
 * Build a vision prompt from ChatMessages.
 * Extracts text with `<__media__>` markers for each image and returns
 * the ordered image buffers.
 */
export async function buildVisionPrompt(
  messages: ChatMessage[],
): Promise<{ text: string; imageBuffers: Buffer[] }> {
  const imageBuffers: Buffer[] = [];
  const textParts: string[] = [];

  for (const msg of messages) {
    if (typeof msg.content === 'string') {
      textParts.push(msg.content);
      continue;
    }

    const msgParts: string[] = [];
    for (const part of msg.content) {
      if (part.type === 'text') {
        msgParts.push(part.text);
      } else if (part.type === 'image_url') {
        const buf = await resolveImageToBuffer(part.image_url.url);
        imageBuffers.push(buf);
        msgParts.push(MEDIA_MARKER);
      }
    }
    textParts.push(msgParts.join(''));
  }

  return { text: textParts.join('\n'), imageBuffers };
}
