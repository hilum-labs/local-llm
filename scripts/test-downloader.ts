import { ModelManager } from '../packages/local-llm/src/model-manager.js';

let passed = 0;
let failed = 0;

function check(label: string, ok: boolean) {
  if (ok) {
    console.log(`${label.padEnd(55)} PASS`);
    passed++;
  } else {
    console.log(`${label.padEnd(55)} FAIL`);
    failed++;
  }
}

async function main() {
  console.log('=== Model Downloader & Cache Manager ===\n');

  const manager = new ModelManager();

  // 1. Download with progress (TinyLlama ~636 MB — already cached if Phase 2 ran)
  const url =
    'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf';

  console.log('Downloading model (or cache hit)...');
  let lastPct = -1;
  const modelPath = await manager.downloadModel(url, {
    onProgress: (_downloaded, _total, pct) => {
      const rounded = Math.floor(pct);
      if (rounded > lastPct) {
        lastPct = rounded;
        process.stdout.write(`\rDownloading: ${rounded}%`);
      }
    },
  });
  if (lastPct >= 0) console.log(); // newline after progress
  console.log(`Model path: ${modelPath}`);
  check('downloadModel() returns a path', modelPath.length > 0);
  check('Path ends with .gguf', modelPath.endsWith('.gguf'));

  // 2. Cache hit — should be instant (no progress callback fired)
  let progressFired = false;
  const t0 = Date.now();
  const cachedPath = await manager.downloadModel(url, {
    onProgress: () => {
      progressFired = true;
    },
  });
  const elapsed = Date.now() - t0;
  check('Cache hit returns same path', cachedPath === modelPath);
  check('Cache hit is fast (< 100ms)', elapsed < 100);
  check('Cache hit skips download', !progressFired);

  // 3. Shorthand URL resolution
  const shorthandPath = await manager.downloadModel(
    'TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
  );
  check('Shorthand resolves to same model', shorthandPath === modelPath);

  // 4. List models
  const models = await manager.listModels();
  check('listModels() returns entries', models.length > 0);
  check('Listed model matches URL', models.some((m) => m.url === url));

  // Summary
  console.log(`\n=== Results: ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
