const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');
const addonPath = path.join(root, 'packages', 'native', 'build', 'Release', 'hilum_native.node');
const modelCandidates = [
  process.argv[2],
  process.env.LOCAL_LLM_SMOKE_MODEL,
  path.join(root, '..', 'hilum-local-llm-engine', 'build-review', 'tinyllamas', 'stories15M-q4_0.gguf'),
];
const modelPath = modelCandidates.find((candidate) => candidate && fs.existsSync(candidate));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

async function main() {
  assert(fs.existsSync(addonPath), `Native addon not found: ${addonPath}`);
  assert(modelPath, 'Pass a GGUF model path or set LOCAL_LLM_SMOKE_MODEL');

  const native = require(addonPath);
  native.setLogCallback(() => {});
  const model = native.loadModel(modelPath, { n_gpu_layers: 0, use_mmap: true });
  const streamContext = native.createContext(model, {
    n_ctx: 128,
    n_batch: 64,
    n_threads: 1,
    n_seq_max: 1,
  });
  const batchContext = native.createContext(model, {
    n_ctx: 128,
    n_batch: 64,
    n_threads: 1,
    n_seq_max: 2,
  });

  const completion = new Promise((resolve, reject) => {
    let tokens = 0;
    native.inferStream(
      model,
      streamContext,
      'Once upon a time',
      { max_tokens: 8, temperature: 0, top_k: 1, seed: 42 },
      (error, token) => {
        if (error) {
          reject(error);
        } else if (token === null) {
          resolve(tokens);
        } else {
          tokens++;
        }
      },
    );
  });

  const batchCompletion = new Promise((resolve, reject) => {
    const completed = new Set();
    const tokenCounts = [0, 0];
    native.inferBatch(
      model,
      batchContext,
      ['The quick brown fox', 'She opened the door and'],
      [{ max_tokens: 2, temperature: 0, top_k: 1, dry_multiplier: 0.5,
        dry_sequence_breakers: ['\n', ':'] }],
      (error, token, sequence, finishReason) => {
        if (error) {
          reject(error);
        } else if (sequence === -1) {
          resolve(tokenCounts);
        } else if (finishReason) {
          completed.add(sequence);
        } else if (token !== null) {
          tokenCounts[sequence]++;
        }
      },
    );
  });

  native.freeContext(streamContext);
  native.freeContext(batchContext);
  native.freeModel(model);
  native.freeContext(streamContext);
  native.freeContext(batchContext);
  native.freeModel(model);

  let streamContextRejected = false;
  let batchContextRejected = false;
  let modelRejected = false;
  try { native.getContextSize(streamContext); } catch { streamContextRejected = true; }
  try { native.getContextSize(batchContext); } catch { batchContextRejected = true; }
  try { native.getModelSize(model); } catch { modelRejected = true; }
  assert(streamContextRejected, 'Released streaming context remained usable');
  assert(batchContextRejected, 'Released batch context remained usable');
  assert(modelRejected, 'Released model remained usable');

  const [streamTokens, batchTokens] = await Promise.all([completion, batchCompletion]);
  assert(streamTokens > 0, 'Streaming inference completed without emitting tokens');
  assert(batchTokens.every((count) => count > 0), 'Batch inference completed without tokens');
  console.log(
    `PASS native lifetime stress (${streamTokens} streamed tokens, ` +
    `${batchTokens.join('/')} batch tokens after immediate release)`,
  );
  native.setLogCallback(null);
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : error);
  process.exit(1);
});
