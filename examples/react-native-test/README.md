# local-llm-rn Test App

Minimal React Native app for testing `local-llm-rn` on iOS.

## Prerequisites

- Xcode 16+
- CocoaPods
- Node.js 20+

## Setup

```bash
# Install dependencies
npm install

# Generate the iOS project (if not already present)
npx @react-native-community/cli init LocalLLMTest --directory . --skip-install

# Install pods (compiles the C++ engine via CocoaPods)
cd ios && pod install && cd ..
```

## Run

```bash
# Start Metro bundler
npm start

# In another terminal — build and run on iOS simulator or device
npm run ios
```

## What it tests

1. **Device capabilities** — RAM, GPU, Metal version, recommended quantization
2. **Model download** — progress indicator, HuggingFace GGUF download
3. **Streaming chat** — real-time token-by-token generation
4. **Memory cleanup** — `dispose()` frees model and context
