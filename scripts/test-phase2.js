#!/usr/bin/env node
/**
 * Phase 2 test script — model loading, tokenization, chat templates, inference
 * Requires a GGUF model file. Downloads TinyLlama 1.1B Q4_K_M if not present.
 */

const fs = require("fs");
const path = require("path");
const https = require("https");
const { execSync } = require("child_process");

const MODELS_DIR = path.join(__dirname, "..", "models");
const MODEL_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";
const MODEL_PATH = path.join(MODELS_DIR, MODEL_FILENAME);
const MODEL_URL =
  "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf";

const native = require("../packages/native/build/Release/hilum_native.node");

let passed = 0;
let failed = 0;

function pass(name, detail) {
  passed++;
  console.log(`  ✅ ${name}: ${detail || "OK"}`);
}

function fail(name, detail) {
  failed++;
  console.error(`  ❌ ${name}: ${detail}`);
}

function assert(name, condition, detail) {
  if (condition) pass(name, detail);
  else fail(name, detail || "assertion failed");
}

// ── Download model if needed ────────────────────────────────────────────────

function downloadModel() {
  return new Promise((resolve, reject) => {
    if (fs.existsSync(MODEL_PATH)) {
      console.log(`  Model already present: ${MODEL_FILENAME}`);
      return resolve();
    }

    fs.mkdirSync(MODELS_DIR, { recursive: true });
    console.log(`  Downloading ${MODEL_FILENAME} (~670 MB)...`);

    const file = fs.createWriteStream(MODEL_PATH + ".tmp");
    const follow = (url) => {
      https
        .get(url, { headers: { "User-Agent": "hilum-test" } }, (res) => {
          if (res.statusCode >= 300 && res.statusCode < 400 && res.headers.location) {
            return follow(res.headers.location);
          }
          if (res.statusCode !== 200) {
            return reject(new Error(`HTTP ${res.statusCode} from ${url}`));
          }

          const total = parseInt(res.headers["content-length"] || "0", 10);
          let downloaded = 0;

          res.on("data", (chunk) => {
            downloaded += chunk.length;
            file.write(chunk);
            if (total > 0) {
              const pct = ((downloaded / total) * 100).toFixed(1);
              process.stdout.write(`\r  Progress: ${pct}% (${(downloaded / 1e6).toFixed(0)} MB)`);
            }
          });

          res.on("end", () => {
            file.end();
            process.stdout.write("\n");
            fs.renameSync(MODEL_PATH + ".tmp", MODEL_PATH);
            console.log(`  Download complete.`);
            resolve();
          });

          res.on("error", reject);
        })
        .on("error", reject);
    };
    follow(MODEL_URL);
  });
}

// ── Tests ───────────────────────────────────────────────────────────────────

async function main() {
  console.log("\n═══ Phase 2: Core Native Bindings Tests ═══\n");

  // Phase 1 sanity check
  console.log("[0] Phase 1 sanity");
  assert("backendInfo", typeof native.backendInfo() === "string", native.backendInfo().slice(0, 60) + "...");
  assert("backendVersion", native.backendVersion().includes("local-llm-native"));

  // Download model
  console.log("\n[1] Model download");
  await downloadModel();

  // Load model
  console.log("\n[2] Load model");
  let model;
  try {
    model = native.loadModel(MODEL_PATH, { n_gpu_layers: 99 });
    pass("loadModel", "TinyLlama 1.1B loaded");
  } catch (e) {
    fail("loadModel", e.message);
    process.exit(1);
  }

  // Create context
  console.log("\n[3] Create context");
  let ctx;
  try {
    ctx = native.createContext(model, { n_ctx: 2048 });
    pass("createContext", "n_ctx=2048");
  } catch (e) {
    fail("createContext", e.message);
    native.freeModel(model);
    process.exit(1);
  }

  // Tokenize / detokenize
  console.log("\n[4] Tokenize / Detokenize");
  try {
    const text = "Hello world";
    const tokens = native.tokenize(model, text, true, false);
    assert("tokenize", tokens.length > 0, `"${text}" → ${tokens.length} tokens [${Array.from(tokens).join(",")}]`);

    const decoded = native.detokenize(model, tokens);
    // BOS token may add leading space or special tokens — trim for comparison
    const matches = decoded.includes("Hello") && decoded.includes("world");
    assert("detokenize round-trip", matches, `"${decoded.trim()}"`);
  } catch (e) {
    fail("tokenize/detokenize", e.message);
  }

  // Chat template
  console.log("\n[5] Chat template");
  try {
    const messages = [
      { role: "user", content: "Say hi in one word." },
    ];
    const formatted = native.applyChatTemplate(model, messages, true);
    assert("applyChatTemplate", formatted.length > 0, `"${formatted.slice(0, 80)}..."`);
  } catch (e) {
    fail("applyChatTemplate", e.message);
  }

  // inferSync
  console.log("\n[6] inferSync");
  try {
    const messages = [
      { role: "user", content: "What is 2+2? Answer with just the number." },
    ];
    const prompt = native.applyChatTemplate(model, messages, true);
    const result = native.inferSync(model, ctx, prompt, {
      max_tokens: 32,
      temperature: 0.1,
      seed: 42,
    });
    assert("inferSync", result.length > 0, `"${result.trim().slice(0, 100)}"`);
  } catch (e) {
    fail("inferSync", e.message);
  }

  // Need a fresh context for streaming (KV cache has prior inference)
  native.freeContext(ctx);
  ctx = native.createContext(model, { n_ctx: 2048 });

  // inferStream
  console.log("\n[7] inferStream");
  await new Promise((resolve) => {
    try {
      const messages = [
        { role: "user", content: "Count from 1 to 5." },
      ];
      const prompt = native.applyChatTemplate(model, messages, true);
      let accumulated = "";
      let tokenCount = 0;

      native.inferStream(model, ctx, prompt, { max_tokens: 32, temperature: 0.1, seed: 42 }, (err, token) => {
        if (err) {
          fail("inferStream", err.message || err);
          return resolve();
        }
        if (token === null) {
          assert("inferStream", tokenCount > 0, `${tokenCount} tokens: "${accumulated.trim().slice(0, 100)}"`);
          return resolve();
        }
        accumulated += token;
        tokenCount++;
        process.stdout.write(token);
      });
    } catch (e) {
      fail("inferStream", e.message);
      resolve();
    }
  });

  // Cleanup
  console.log("\n\n[8] Cleanup");
  try {
    native.freeContext(ctx);
    native.freeModel(model);
    pass("cleanup", "context and model freed");
  } catch (e) {
    fail("cleanup", e.message);
  }

  // Summary
  console.log(`\n═══ Results: ${passed} passed, ${failed} failed ═══\n`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((e) => {
  console.error("Fatal:", e);
  process.exit(1);
});
