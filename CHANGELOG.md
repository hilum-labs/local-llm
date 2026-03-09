# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-03-02

### Added

- OpenAI-compatible `chat.completions.create()` API (streaming and non-streaming)
- Vision / multimodal support with GPT-4V content format (images via data URIs, file paths, or URLs)
- Vercel AI SDK `LanguageModelV3` provider for `generateText()` and `streamText()`
- Auto model download from HuggingFace (URL or shorthand `user/repo/file.gguf`)
- Model caching with configurable `cacheDir` and download progress callback
- GPU auto-detection (Metal on macOS, CUDA on Linux/Windows)
- Compute modes: `auto`, `gpu`, `cpu`, `hybrid`
- Model pooling with LRU eviction and memory limits (`LocalLLM.pool`)
- Model preloading via `LocalLLM.preload()` for fast startup
- Lower-level engine API: `Model`, `InferenceContext`, `ModelManager`
- Platform support: macOS (Apple Silicon, Intel), Linux x64, Windows x64
- TypeScript types and ESM build

[1.0.0]: https://github.com/hilum-labs/local-llm/releases/tag/v1.0.0
