# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Retain models, inference contexts, embedding contexts, and vision projectors
  until detached native workers complete, preventing use-after-free crashes.
- Use the dedicated embedding-context destructor from the shared native contract.
- Correct advanced sampler option mapping against Hilum engine API 1.0.1.

### Changed

- Added real-model native lifetime regression coverage and stronger binary smoke checks.
- Upgraded the native build toolchain and locked dependency graph with no known vulnerabilities.

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
