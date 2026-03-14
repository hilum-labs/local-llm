const fs = require("fs");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const ADDON_PATH = path.join(ROOT, "packages", "native", "build", "Release", "hilum_native.node");
const DEFAULT_MODEL_PATH = path.join(ROOT, "models", "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf");

let passed = 0;
let failed = 0;

function pass(name, detail) {
  passed++;
  console.log(`PASS  ${name}${detail ? ` -- ${detail}` : ""}`);
}

function fail(name, detail) {
  failed++;
  console.error(`FAIL  ${name}${detail ? ` -- ${detail}` : ""}`);
}

function assert(name, condition, detail) {
  if (condition) {
    pass(name, detail);
  } else {
    fail(name, detail || "assertion failed");
  }
}

function modelPath() {
  return process.env.LOCAL_LLM_SMOKE_MODEL || process.env.LOCAL_LLM_PHASE0_MODEL || DEFAULT_MODEL_PATH;
}

async function main() {
  console.log("=== local-llm native smoke ===");

  assert("native addon exists", fs.existsSync(ADDON_PATH), ADDON_PATH);
  assert("default model exists", fs.existsSync(modelPath()), modelPath());
  if (!fs.existsSync(ADDON_PATH) || !fs.existsSync(modelPath())) {
    process.exit(1);
  }

  const native = require(ADDON_PATH);
  const expectedExports = [
    "apiVersion",
    "applyChatTemplate",
    "backendInfo",
    "backendVersion",
    "createContext",
    "detokenize",
    "freeContext",
    "freeModel",
    "inferStream",
    "inferSync",
    "jsonSchemaToGrammar",
    "loadModel",
    "tokenize",
  ];

  for (const key of expectedExports) {
    assert(`addon exports ${key}`, typeof native[key] === "function");
  }

  const backendVersion = String(native.backendVersion());
  assert("backend version is non-empty", backendVersion.length > 0);

  let model = null;
  let ctx = null;

  try {
    model = native.loadModel(modelPath(), { n_gpu_layers: 0, use_mmap: true });
    assert("loadModel", Boolean(model));

    ctx = native.createContext(model, { n_ctx: 1024, n_batch: 128, n_threads: 1 });
    assert("createContext", Boolean(ctx));

    const tokens = native.tokenize(model, "Hello from local-llm", true, false);
    assert("tokenize returns tokens", Array.from(tokens).length > 0);

    const detok = native.detokenize(model, tokens);
    assert("detokenize returns text", typeof detok === "string" && detok.includes("Hello"));

    const prompt = native.applyChatTemplate(model, [
      { role: "user", content: "Reply with exactly: hello world" },
    ], true);
    assert("applyChatTemplate returns a prompt", typeof prompt === "string" && prompt.length > 0);

    const grammar = native.jsonSchemaToGrammar(JSON.stringify({
      type: "object",
      properties: {
        answer: { type: "string" },
      },
      required: ["answer"],
    }));
    assert("jsonSchemaToGrammar returns grammar", typeof grammar === "string" && grammar.includes("root"));

    const syncText = native.inferSync(model, ctx, prompt, {
      max_tokens: 24,
      temperature: 0.1,
      seed: 42,
    });
    assert("inferSync returns text", typeof syncText === "string" && syncText.length > 0, syncText.slice(0, 80));

    native.freeContext(ctx);
    ctx = native.createContext(model, { n_ctx: 1024, n_batch: 128, n_threads: 1 });

    const streamText = await new Promise((resolve, reject) => {
      let output = "";
      let tokensSeen = 0;

      native.inferStream(model, ctx, prompt, {
        max_tokens: 16,
        temperature: 0.1,
        seed: 42,
      }, (err, token) => {
        if (err) {
          reject(err);
          return;
        }

        if (token === null) {
          if (tokensSeen === 0) {
            reject(new Error("stream completed without tokens"));
            return;
          }
          resolve(output);
          return;
        }

        output += token;
        tokensSeen++;
      });
    });

    assert("inferStream returns text", typeof streamText === "string" && streamText.length > 0, streamText.slice(0, 80));
  } catch (error) {
    fail("native smoke", error && error.message ? error.message : String(error));
  } finally {
    if (ctx) {
      try {
        native.freeContext(ctx);
      } catch (_) {}
    }
    if (model) {
      try {
        native.freeModel(model);
      } catch (_) {}
    }
  }

  console.log(`=== ${passed} passed, ${failed} failed ===`);
  process.exit(failed > 0 ? 1 : 0);
}

main().catch((error) => {
  console.error("Fatal:", error);
  process.exit(1);
});
