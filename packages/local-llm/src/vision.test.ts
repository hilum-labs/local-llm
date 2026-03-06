import { describe, it, expect } from 'vitest';
import { resolveImageToBuffer, buildVisionPrompt, hasImageContent } from './engine.js';

describe('resolveImageToBuffer', () => {
  it('decodes base64 data URI', async () => {
    const pngBytes = Buffer.from([0x89, 0x50, 0x4e, 0x47]); // PNG magic bytes
    const dataUri = `data:image/png;base64,${pngBytes.toString('base64')}`;
    const result = await resolveImageToBuffer(dataUri);
    expect(Buffer.compare(result, pngBytes)).toBe(0);
  });

  it('decodes base64 data URI with jpeg mime type', async () => {
    const jpegBytes = Buffer.from([0xff, 0xd8, 0xff, 0xe0]); // JPEG magic bytes
    const dataUri = `data:image/jpeg;base64,${jpegBytes.toString('base64')}`;
    const result = await resolveImageToBuffer(dataUri);
    expect(Buffer.compare(result, jpegBytes)).toBe(0);
  });

  it('throws on invalid data URI (no comma)', async () => {
    await expect(resolveImageToBuffer('data:image/png;base64')).rejects.toThrow('Invalid data URI');
  });

  it('reads local file path', async () => {
    // Use this test file itself as a readable file
    const result = await resolveImageToBuffer(import.meta.filename);
    expect(result.length).toBeGreaterThan(0);
  });
});

describe('buildVisionPrompt', () => {
  it('returns plain text for text-only messages', async () => {
    const result = await buildVisionPrompt([
      { role: 'user', content: 'Hello world' },
    ]);
    expect(result.text).toBe('Hello world');
    expect(result.imageBuffers).toHaveLength(0);
  });

  it('inserts media markers and collects image buffers', async () => {
    const imageBytes = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
    const dataUri = `data:image/png;base64,${imageBytes.toString('base64')}`;

    const result = await buildVisionPrompt([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'What is in this image?' },
          { type: 'image_url', image_url: { url: dataUri } },
        ],
      },
    ]);

    expect(result.text).toBe('What is in this image?<__media__>');
    expect(result.imageBuffers).toHaveLength(1);
    expect(Buffer.compare(result.imageBuffers[0], imageBytes)).toBe(0);
  });

  it('handles multiple images in one message', async () => {
    const img1 = Buffer.from([1, 2, 3]);
    const img2 = Buffer.from([4, 5, 6]);
    const uri1 = `data:image/png;base64,${img1.toString('base64')}`;
    const uri2 = `data:image/png;base64,${img2.toString('base64')}`;

    const result = await buildVisionPrompt([
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Compare these:' },
          { type: 'image_url', image_url: { url: uri1 } },
          { type: 'text', text: ' and ' },
          { type: 'image_url', image_url: { url: uri2 } },
        ],
      },
    ]);

    expect(result.text).toBe('Compare these:<__media__> and <__media__>');
    expect(result.imageBuffers).toHaveLength(2);
  });

  it('handles mixed text-only and multimodal messages', async () => {
    const img = Buffer.from([0x89, 0x50]);
    const uri = `data:image/png;base64,${img.toString('base64')}`;

    const result = await buildVisionPrompt([
      { role: 'system', content: 'You are helpful.' },
      {
        role: 'user',
        content: [
          { type: 'text', text: 'Describe:' },
          { type: 'image_url', image_url: { url: uri } },
        ],
      },
    ]);

    expect(result.text).toBe('You are helpful.\nDescribe:<__media__>');
    expect(result.imageBuffers).toHaveLength(1);
  });
});

describe('hasImageContent', () => {
  it('returns false for text-only messages', () => {
    expect(hasImageContent([
      { content: 'Hello' },
      { content: 'World' },
    ])).toBe(false);
  });

  it('returns true when image_url part exists', () => {
    expect(hasImageContent([
      {
        content: [
          { type: 'text', text: 'Hello' },
          { type: 'image_url', image_url: { url: 'data:...' } },
        ],
      },
    ])).toBe(true);
  });

  it('returns false for array content with only text parts', () => {
    expect(hasImageContent([
      {
        content: [
          { type: 'text', text: 'Hello' },
          { type: 'text', text: 'World' },
        ],
      },
    ])).toBe(false);
  });
});
