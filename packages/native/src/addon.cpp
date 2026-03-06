#include <napi.h>
#include <thread>
#include <string>
#include <vector>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "json-schema-to-grammar.h"
#include <nlohmann/json.hpp>

// ── Backend initialization (once) ───────────────────────────────────────────

static bool g_backend_initialized = false;

static void ensure_backend() {
    if (!g_backend_initialized) {
        llama_backend_init();
        g_backend_initialized = true;
    }
}

// ── Live handle registry (double-free protection) ───────────────────────────
// Tracks all allocated native handles so free functions are idempotent.

static std::unordered_set<void*> g_live_handles;

static void register_handle(void* ptr) { g_live_handles.insert(ptr); }
static bool unregister_handle(void* ptr) { return g_live_handles.erase(ptr) > 0; }

// Forward declarations
static void evict_sampler_cache(llama_context* ctx);

// ── Helpers ─────────────────────────────────────────────────────────────────

// Thread-local buffer for token_to_piece — avoids per-token heap allocations.
// Most tokens fit in 256 bytes; oversized tokens fall back to the vector.
struct TokenBuf {
    char small[256];
    std::vector<char> big;
    const char* ptr;
    int32_t len;

    void decode(const llama_vocab* vocab, llama_token token) {
        int32_t n = llama_token_to_piece(vocab, token, small, sizeof(small), 0, false);
        if (n >= 0) {
            ptr = small;
            len = n;
        } else {
            big.resize(-n);
            llama_token_to_piece(vocab, token, big.data(), big.size(), 0, false);
            ptr = big.data();
            len = -n;
        }
    }
};

static thread_local TokenBuf t_token_buf;

// Convenience wrapper that returns std::string (used where a copy is needed)
static std::string token_to_piece(const llama_vocab* vocab, llama_token token) {
    t_token_buf.decode(vocab, token);
    return std::string(t_token_buf.ptr, t_token_buf.len);
}

// Append token directly to a string without intermediate std::string construction
static void token_append(const llama_vocab* vocab, llama_token token, std::string& out) {
    t_token_buf.decode(vocab, token);
    out.append(t_token_buf.ptr, t_token_buf.len);
}

// ── Backend info (Phase 1) ──────────────────────────────────────────────────

Napi::Value BackendInfo(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return Napi::String::New(env, llama_print_system_info());
}

Napi::Value BackendVersion(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    return Napi::String::New(env, "local-llm-native v1.0.0 (hilum-local-llm-engine)");
}

// ── Resource Management ─────────────────────────────────────────────────────

Napi::Value LoadModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "loadModel: first argument must be a string path")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    ensure_backend();

    std::string path = info[0].As<Napi::String>().Utf8Value();
    llama_model_params params = llama_model_default_params();

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("n_gpu_layers"))
            params.n_gpu_layers = opts.Get("n_gpu_layers").As<Napi::Number>().Int32Value();
        if (opts.Has("use_mmap"))
            params.use_mmap = opts.Get("use_mmap").As<Napi::Boolean>().Value();
    }

    llama_model* model = llama_model_load_from_file(path.c_str(), params);
    if (!model) {
        Napi::Error::New(env, "loadModel: failed to load model from " + path)
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    register_handle(model);
    return Napi::External<llama_model>::New(env, model);
}

Napi::Value LoadModelFromBuffer(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsBuffer()) {
        Napi::TypeError::New(env, "First argument must be a Buffer").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    ensure_backend();

    Napi::Buffer<uint8_t> buf = info[0].As<Napi::Buffer<uint8_t>>();
    llama_model_params params = llama_model_default_params();
    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("n_gpu_layers"))
            params.n_gpu_layers = opts.Get("n_gpu_layers").As<Napi::Number>().Int32Value();
        if (opts.Has("use_mmap"))
            params.use_mmap = opts.Get("use_mmap").As<Napi::Boolean>().Value();
    }
    params.use_mmap = false;  // force: buffer loading can't use mmap

    llama_model* model = llama_model_load_from_buffer(buf.Data(), buf.Length(), params);
    if (!model) {
        Napi::Error::New(env, "Failed to load model from buffer").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Hold persistent ref to JS Buffer so GC won't collect it while model is alive
    register_handle(model);
    auto* ref = new Napi::Reference<Napi::Buffer<uint8_t>>(Napi::Persistent(buf));
    return Napi::External<llama_model>::New(env, model,
        [ref](Napi::Env, llama_model* m) { unregister_handle(m); llama_model_free(m); delete ref; });
}

Napi::Value GetModelSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "getModelSize: first argument must be a model external")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    return Napi::Number::New(env, static_cast<double>(llama_model_size(model)));
}

Napi::Value CreateContext(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "createContext: first argument must be a model external")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    llama_context_params params = llama_context_default_params();

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("n_ctx"))
            params.n_ctx = opts.Get("n_ctx").As<Napi::Number>().Uint32Value();
        if (opts.Has("n_batch"))
            params.n_batch = opts.Get("n_batch").As<Napi::Number>().Uint32Value();
        if (opts.Has("n_threads"))
            params.n_threads = opts.Get("n_threads").As<Napi::Number>().Int32Value();
        if (opts.Has("flash_attn_type"))
            params.flash_attn_type = static_cast<llama_flash_attn_type>(opts.Get("flash_attn_type").As<Napi::Number>().Int32Value());
        if (opts.Has("type_k"))
            params.type_k = static_cast<ggml_type>(opts.Get("type_k").As<Napi::Number>().Int32Value());
        if (opts.Has("type_v"))
            params.type_v = static_cast<ggml_type>(opts.Get("type_v").As<Napi::Number>().Int32Value());
        if (opts.Has("n_seq_max"))
            params.n_seq_max = opts.Get("n_seq_max").As<Napi::Number>().Uint32Value();
    }

    llama_context* ctx = llama_init_from_model(model, params);
    if (!ctx) {
        Napi::Error::New(env, "createContext: failed to create context")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    register_handle(ctx);
    return Napi::External<llama_context>::New(env, ctx);
}

Napi::Value FreeModel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() >= 1 && info[0].IsExternal()) {
        llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
        if (unregister_handle(model)) {
            llama_model_free(model);
        }
    }
    return env.Undefined();
}

Napi::Value FreeContext(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() >= 1 && info[0].IsExternal()) {
        llama_context* ctx = info[0].As<Napi::External<llama_context>>().Data();
        if (unregister_handle(ctx)) {
            evict_sampler_cache(ctx);
            llama_free(ctx);
        }
    }
    return env.Undefined();
}

// ── Tokenization ────────────────────────────────────────────────────────────

Napi::Value Tokenize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsString()) {
        Napi::TypeError::New(env, "tokenize(model, text, [addSpecial], [parseSpecial])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    std::string text = info[1].As<Napi::String>().Utf8Value();
    bool add_special = (info.Length() >= 3 && info[2].IsBoolean()) ? info[2].As<Napi::Boolean>().Value() : true;
    bool parse_special = (info.Length() >= 4 && info[3].IsBoolean()) ? info[3].As<Napi::Boolean>().Value() : false;

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // First call to get required size
    int32_t n = llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_special, parse_special);
    if (n >= 0) {
        // Shouldn't happen with nullptr/0, but handle gracefully
        return Napi::Int32Array::New(env, 0);
    }

    int32_t n_tokens = -n;
    std::vector<llama_token> tokens(n_tokens);
    int32_t result = llama_tokenize(vocab, text.c_str(), text.size(), tokens.data(), n_tokens, add_special, parse_special);

    if (result < 0) {
        Napi::Error::New(env, "tokenize: failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Int32Array arr = Napi::Int32Array::New(env, result);
    for (int32_t i = 0; i < result; i++) {
        arr[i] = tokens[i];
    }
    return arr;
}

Napi::Value Detokenize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "detokenize(model, tokens)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Accept Int32Array or regular Array
    std::vector<llama_token> tokens;
    if (info[1].IsTypedArray()) {
        Napi::Int32Array arr = info[1].As<Napi::Int32Array>();
        tokens.resize(arr.ElementLength());
        for (size_t i = 0; i < arr.ElementLength(); i++) {
            tokens[i] = arr[i];
        }
    } else if (info[1].IsArray()) {
        Napi::Array arr = info[1].As<Napi::Array>();
        tokens.resize(arr.Length());
        for (uint32_t i = 0; i < arr.Length(); i++) {
            tokens[i] = arr.Get(i).As<Napi::Number>().Int32Value();
        }
    } else {
        Napi::TypeError::New(env, "detokenize: tokens must be Int32Array or Array")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Detokenize with a generous buffer
    std::vector<char> buf(tokens.size() * 16 + 256);
    int32_t n = llama_detokenize(vocab, tokens.data(), tokens.size(),
                                  buf.data(), buf.size(), false, true);
    if (n < 0) {
        // Need bigger buffer
        buf.resize(-n);
        n = llama_detokenize(vocab, tokens.data(), tokens.size(),
                              buf.data(), buf.size(), false, true);
    }

    return Napi::String::New(env, buf.data(), n > 0 ? n : 0);
}

// ── Chat Template ───────────────────────────────────────────────────────────

Napi::Value ApplyChatTemplate(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsArray()) {
        Napi::TypeError::New(env, "applyChatTemplate(model, messages, [addAssistant])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    Napi::Array msgs_arr = info[1].As<Napi::Array>();
    bool add_assistant = (info.Length() >= 3 && info[2].IsBoolean()) ? info[2].As<Napi::Boolean>().Value() : true;

    const char* tmpl = llama_model_chat_template(model, nullptr);

    // Build message array — keep strings alive
    uint32_t n_msgs = msgs_arr.Length();
    std::vector<std::string> roles(n_msgs), contents(n_msgs);
    std::vector<llama_chat_message> messages(n_msgs);

    for (uint32_t i = 0; i < n_msgs; i++) {
        Napi::Object msg = msgs_arr.Get(i).As<Napi::Object>();
        roles[i] = msg.Get("role").As<Napi::String>().Utf8Value();
        contents[i] = msg.Get("content").As<Napi::String>().Utf8Value();
        messages[i].role = roles[i].c_str();
        messages[i].content = contents[i].c_str();
    }

    // First call to get size
    int32_t len = llama_chat_apply_template(tmpl, messages.data(), n_msgs, add_assistant, nullptr, 0);
    if (len < 0) {
        Napi::Error::New(env, "applyChatTemplate: failed to apply template")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::vector<char> buf(len + 1);
    llama_chat_apply_template(tmpl, messages.data(), n_msgs, add_assistant, buf.data(), buf.size());

    return Napi::String::New(env, buf.data(), len);
}

// ── Embeddings ───────────────────────────────────────────────────────────────

Napi::Value GetEmbeddingDimension(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "getEmbeddingDimension(model)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    return Napi::Number::New(env, llama_model_n_embd(model));
}

Napi::Value CreateEmbeddingContext(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "createEmbeddingContext(model, opts?)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    llama_context_params params = llama_context_default_params();
    params.embeddings = true;
    params.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("n_ctx"))
            params.n_ctx = opts.Get("n_ctx").As<Napi::Number>().Uint32Value();
        if (opts.Has("n_batch"))
            params.n_batch = opts.Get("n_batch").As<Napi::Number>().Uint32Value();
        if (opts.Has("n_threads"))
            params.n_threads = opts.Get("n_threads").As<Napi::Number>().Int32Value();
        if (opts.Has("pooling_type"))
            params.pooling_type = static_cast<enum llama_pooling_type>(
                opts.Get("pooling_type").As<Napi::Number>().Int32Value());
    }

    llama_context* ctx = llama_init_from_model(model, params);
    if (!ctx) {
        Napi::Error::New(env, "createEmbeddingContext: failed to create context")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    register_handle(ctx);
    return Napi::External<llama_context>::New(env, ctx);
}

Napi::Value Embed(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3 || !info[0].IsExternal() || !info[1].IsExternal() || !info[2].IsTypedArray()) {
        Napi::TypeError::New(env, "embed(ctx, model, Int32Array tokens)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_context* ctx = info[0].As<Napi::External<llama_context>>().Data();
    llama_model* model = info[1].As<Napi::External<llama_model>>().Data();
    Napi::Int32Array tokens = info[2].As<Napi::Int32Array>();

    int32_t n_tokens = static_cast<int32_t>(tokens.ElementLength());
    int32_t n_embd = llama_model_n_embd(model);

    if (n_tokens == 0) {
        return Napi::Float32Array::New(env, 0);
    }

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]    = tokens[i];
        batch.pos[i]      = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i]   = 1;
    }
    batch.n_tokens = n_tokens;

    // Clear KV cache before encoding a fresh sequence
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) llama_memory_seq_rm(mem, 0, 0, -1);

    int32_t ret = llama_encode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        Napi::Error::New(env, "embed: llama_encode failed (code " + std::to_string(ret) + ")")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    const float* emb = llama_get_embeddings_seq(ctx, 0);
    if (!emb) {
        Napi::Error::New(env, "embed: llama_get_embeddings_seq returned null — "
                              "ensure the model supports embeddings and pooling_type != NONE")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // L2-normalize the embedding
    float norm = 0.0f;
    for (int32_t i = 0; i < n_embd; i++) {
        norm += emb[i] * emb[i];
    }
    norm = std::sqrt(norm);

    Napi::Float32Array result = Napi::Float32Array::New(env, n_embd);
    if (norm > 0.0f) {
        for (int32_t i = 0; i < n_embd; i++) {
            result[i] = emb[i] / norm;
        }
    } else {
        for (int32_t i = 0; i < n_embd; i++) {
            result[i] = 0.0f;
        }
    }

    return result;
}

Napi::Value EmbedBatch(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3 || !info[0].IsExternal() || !info[1].IsExternal() || !info[2].IsArray()) {
        Napi::TypeError::New(env, "embedBatch(ctx, model, Array<Int32Array> tokenArrays)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_context* ctx = info[0].As<Napi::External<llama_context>>().Data();
    llama_model* model = info[1].As<Napi::External<llama_model>>().Data();
    Napi::Array tokenArrays = info[2].As<Napi::Array>();

    int32_t n_embd = llama_model_n_embd(model);
    uint32_t n_inputs = tokenArrays.Length();

    if (n_inputs == 0) {
        return Napi::Array::New(env, 0);
    }

    // Count total tokens across all inputs
    int32_t total_tokens = 0;
    std::vector<Napi::Int32Array> inputs;
    inputs.reserve(n_inputs);
    for (uint32_t i = 0; i < n_inputs; i++) {
        Napi::Int32Array arr = tokenArrays.Get(i).As<Napi::Int32Array>();
        inputs.push_back(arr);
        total_tokens += static_cast<int32_t>(arr.ElementLength());
    }

    llama_batch batch = llama_batch_init(total_tokens, 0, n_inputs);
    int32_t pos_offset = 0;

    for (uint32_t seq = 0; seq < n_inputs; seq++) {
        int32_t n_tok = static_cast<int32_t>(inputs[seq].ElementLength());
        for (int32_t j = 0; j < n_tok; j++) {
            int32_t idx = pos_offset + j;
            batch.token[idx]      = inputs[seq][j];
            batch.pos[idx]        = j;
            batch.n_seq_id[idx]   = 1;
            batch.seq_id[idx][0]  = static_cast<llama_seq_id>(seq);
            batch.logits[idx]     = 1;
        }
        pos_offset += n_tok;
    }
    batch.n_tokens = total_tokens;

    // Clear KV cache
    llama_memory_t mem = llama_get_memory(ctx);
    if (mem) {
        for (uint32_t seq = 0; seq < n_inputs; seq++) {
            llama_memory_seq_rm(mem, static_cast<llama_seq_id>(seq), 0, -1);
        }
    }

    int32_t ret = llama_encode(ctx, batch);
    llama_batch_free(batch);

    if (ret != 0) {
        Napi::Error::New(env, "embedBatch: llama_encode failed (code " + std::to_string(ret) + ")")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array results = Napi::Array::New(env, n_inputs);

    for (uint32_t seq = 0; seq < n_inputs; seq++) {
        const float* emb = llama_get_embeddings_seq(ctx, static_cast<llama_seq_id>(seq));

        Napi::Float32Array vec = Napi::Float32Array::New(env, n_embd);
        if (emb) {
            float norm = 0.0f;
            for (int32_t i = 0; i < n_embd; i++) {
                norm += emb[i] * emb[i];
            }
            norm = std::sqrt(norm);
            if (norm > 0.0f) {
                for (int32_t i = 0; i < n_embd; i++) {
                    vec[i] = emb[i] / norm;
                }
            }
        }
        results.Set(seq, vec);
    }

    return results;
}

// ── Context Info ─────────────────────────────────────────────────────────────

Napi::Value GetContextSize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "getContextSize(ctx)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    llama_context* ctx = info[0].As<Napi::External<llama_context>>().Data();
    return Napi::Number::New(env, static_cast<double>(llama_n_ctx(ctx)));
}

// ── KV Cache Management ─────────────────────────────────────────────────────

Napi::Value KvCacheClear(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "kvCacheClear(ctx, fromPos)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_context* ctx = info[0].As<Napi::External<llama_context>>().Data();
    int32_t from_pos = info[1].As<Napi::Number>().Int32Value();

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, 0, from_pos, -1);

    return env.Undefined();
}

// ── Sampling helpers ────────────────────────────────────────────────────────

static void parse_gen_options(const Napi::Object& opts,
    int32_t& max_tokens, float& temperature, float& top_p,
    int32_t& top_k, float& repeat_penalty, float& frequency_penalty,
    float& presence_penalty, uint32_t& seed,
    std::string& grammar, std::string& grammar_root) {
    if (opts.Has("max_tokens"))         max_tokens = opts.Get("max_tokens").As<Napi::Number>().Int32Value();
    if (opts.Has("temperature"))        temperature = opts.Get("temperature").As<Napi::Number>().FloatValue();
    if (opts.Has("top_p"))              top_p = opts.Get("top_p").As<Napi::Number>().FloatValue();
    if (opts.Has("top_k"))              top_k = opts.Get("top_k").As<Napi::Number>().Int32Value();
    if (opts.Has("repeat_penalty"))     repeat_penalty = opts.Get("repeat_penalty").As<Napi::Number>().FloatValue();
    if (opts.Has("frequency_penalty"))  frequency_penalty = opts.Get("frequency_penalty").As<Napi::Number>().FloatValue();
    if (opts.Has("presence_penalty"))   presence_penalty = opts.Get("presence_penalty").As<Napi::Number>().FloatValue();
    if (opts.Has("seed"))               seed = opts.Get("seed").As<Napi::Number>().Uint32Value();
    if (opts.Has("grammar"))            grammar = opts.Get("grammar").As<Napi::String>().Utf8Value();
    if (opts.Has("grammar_root"))       grammar_root = opts.Get("grammar_root").As<Napi::String>().Utf8Value();
}

// ── Sampler cache ───────────────────────────────────────────────────────────
// Caches the last sampler chain per-context when grammar is not used.
// Grammar samplers accumulate state and can't be safely reused across calls.

struct SamplerParams {
    int32_t  top_k              = 40;
    float    top_p              = 0.9f;
    float    temperature        = 0.7f;
    float    repeat_penalty     = 1.1f;
    float    frequency_penalty  = 0.0f;
    float    presence_penalty   = 0.0f;
    uint32_t seed               = LLAMA_DEFAULT_SEED;

    bool operator==(const SamplerParams& o) const {
        return top_k == o.top_k && top_p == o.top_p &&
               temperature == o.temperature &&
               repeat_penalty == o.repeat_penalty &&
               frequency_penalty == o.frequency_penalty &&
               presence_penalty == o.presence_penalty &&
               seed == o.seed;
    }
};

struct SamplerCache {
    llama_sampler* sampler = nullptr;
    SamplerParams  params;

    void clear() {
        if (sampler) { llama_sampler_free(sampler); sampler = nullptr; }
    }
};

static std::unordered_map<llama_context*, SamplerCache> g_sampler_cache;

static llama_sampler* create_sampler(const llama_vocab* vocab, int32_t top_k,
    float top_p, float temperature, float repeat_penalty,
    float frequency_penalty, float presence_penalty, uint32_t seed,
    const std::string& grammar_str = "", const std::string& grammar_root = "root",
    llama_context* cache_ctx = nullptr) {

    // Try to reuse cached sampler (only when no grammar — grammar is stateful)
    if (cache_ctx && grammar_str.empty()) {
        SamplerParams cur{top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed};
        auto it = g_sampler_cache.find(cache_ctx);
        if (it != g_sampler_cache.end() && it->second.sampler && it->second.params == cur) {
            llama_sampler* cached = it->second.sampler;
            it->second.sampler = nullptr;
            return cached;
        }
    }

    llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
    llama_sampler* smpl = llama_sampler_chain_init(chain_params);

    if (!grammar_str.empty()) {
        llama_sampler* grammar_smpl = llama_sampler_init_grammar(vocab, grammar_str.c_str(), grammar_root.c_str());
        if (grammar_smpl) {
            llama_sampler_chain_add(smpl, grammar_smpl);
        }
    }

    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_penalties(64, repeat_penalty, frequency_penalty, presence_penalty));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    return smpl;
}

// Return a sampler to the cache instead of freeing it (no-grammar calls only)
static void return_sampler(llama_context* cache_ctx, llama_sampler* smpl,
    const SamplerParams& params, bool has_grammar) {
    if (!cache_ctx || has_grammar) {
        llama_sampler_free(smpl);
        return;
    }
    auto& entry = g_sampler_cache[cache_ctx];
    entry.clear();  // free any existing cached sampler
    entry.sampler = smpl;
    entry.params = params;
}

// Clean up cached sampler when a context is freed
static void evict_sampler_cache(llama_context* ctx) {
    auto it = g_sampler_cache.find(ctx);
    if (it != g_sampler_cache.end()) {
        it->second.clear();
        g_sampler_cache.erase(it);
    }
}

// ── Inference — Synchronous ─────────────────────────────────────────────────

Napi::Value InferSync(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 3 || !info[0].IsExternal() || !info[1].IsExternal() || !info[2].IsString()) {
        Napi::TypeError::New(env, "inferSync(model, ctx, prompt, [options])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    llama_context* ctx = info[1].As<Napi::External<llama_context>>().Data();
    std::string prompt = info[2].As<Napi::String>().Utf8Value();

    // Defaults
    int32_t max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    float repeat_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    std::string grammar_str;
    std::string grammar_root = "root";

    if (info.Length() >= 4 && info[3].IsObject()) {
        parse_gen_options(info[3].As<Napi::Object>(),
            max_tokens, temperature, top_p, top_k, repeat_penalty,
            frequency_penalty, presence_penalty, seed, grammar_str, grammar_root);
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // llama_tokenize returns negative required size when buffer is too small
    int32_t n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, false);
    if (n_prompt >= 0) n_prompt = 0;
    else n_prompt = -n_prompt;

    std::vector<llama_token> prompt_tokens(n_prompt);
    llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), n_prompt, true, false);

    // Create sampler chain (with optional grammar constraint, cached when possible)
    SamplerParams smpl_params{top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed};
    llama_sampler* smpl = create_sampler(vocab, top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed, grammar_str, grammar_root, ctx);

    // Prompt eval
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), n_prompt);
    if (llama_decode(ctx, batch) != 0) {
        return_sampler(ctx, smpl, smpl_params, !grammar_str.empty());
        Napi::Error::New(env, "inferSync: prompt eval failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Generation loop
    std::string result;
    llama_token eos = llama_vocab_eos(vocab);

    for (int32_t i = 0; i < max_tokens; i++) {
        llama_token token = llama_sampler_sample(smpl, ctx, -1);

        if (token == eos) break;

        token_append(vocab, token, result);
        llama_sampler_accept(smpl, token);

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, next) != 0) break;
    }

    return_sampler(ctx, smpl, smpl_params, !grammar_str.empty());
    return Napi::String::New(env, result);
}

// ── Inference — Streaming (Async) ───────────────────────────────────────────

struct StreamData {
    std::string token;
    bool done;
    std::string error;
};

void InferStreamCallback(Napi::Env env, Napi::Function jsCallback, void* /*context*/, StreamData* data) {
    if (data->error.size() > 0) {
        jsCallback.Call({
            Napi::Error::New(env, data->error).Value(),
            env.Null()
        });
    } else if (data->done) {
        jsCallback.Call({env.Null(), env.Null()});
    } else {
        jsCallback.Call({env.Null(), Napi::String::New(env, data->token)});
    }
    delete data;
}

using TSFN = Napi::TypedThreadSafeFunction<void, StreamData, InferStreamCallback>;

Napi::Value InferStream(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 5 || !info[0].IsExternal() || !info[1].IsExternal() ||
        !info[2].IsString() || !info[4].IsFunction()) {
        Napi::TypeError::New(env, "inferStream(model, ctx, prompt, options, callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    llama_context* ctx = info[1].As<Napi::External<llama_context>>().Data();
    std::string prompt = info[2].As<Napi::String>().Utf8Value();

    int32_t max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    float repeat_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    std::string grammar_str;
    std::string grammar_root = "root";
    int32_t n_past = 0;

    if (info[3].IsObject()) {
        Napi::Object opts = info[3].As<Napi::Object>();
        parse_gen_options(opts, max_tokens, temperature, top_p, top_k,
            repeat_penalty, frequency_penalty, presence_penalty, seed,
            grammar_str, grammar_root);
        if (opts.Has("n_past")) n_past = opts.Get("n_past").As<Napi::Number>().Int32Value();
    }

    TSFN tsfn = TSFN::New(env, info[4].As<Napi::Function>(), "InferStream", 0, 1);

    std::thread([=]() mutable {
        const llama_vocab* vocab = llama_model_get_vocab(model);

        // llama_tokenize returns negative required size when buffer is too small
        int32_t n_prompt = llama_tokenize(vocab, prompt.c_str(), prompt.size(), nullptr, 0, true, false);
        if (n_prompt >= 0) n_prompt = 0;
        else n_prompt = -n_prompt;

        std::vector<llama_token> prompt_tokens(n_prompt);
        llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), n_prompt, true, false);

        // Sampler chain (with optional grammar constraint, cached when possible)
        SamplerParams smpl_params{top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed};
        llama_sampler* smpl = create_sampler(vocab, top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed, grammar_str, grammar_root, ctx);
        bool has_grammar = !grammar_str.empty();

        // Prompt eval — skip tokens already in KV cache (n_past)
        int32_t eval_start = (n_past > 0 && n_past <= n_prompt) ? n_past : 0;
        int32_t eval_count = n_prompt - eval_start;

        if (eval_count > 0) {
            llama_batch batch = llama_batch_get_one(prompt_tokens.data() + eval_start, eval_count);
            if (llama_decode(ctx, batch) != 0) {
                auto* data = new StreamData{"", false, "inferStream: prompt eval failed"};
                tsfn.BlockingCall(data);
                tsfn.Release();
                return_sampler(ctx, smpl, smpl_params, has_grammar);
                return;
            }
        }

        // Generation loop
        llama_token eos = llama_vocab_eos(vocab);

        for (int32_t i = 0; i < max_tokens; i++) {
            llama_token token = llama_sampler_sample(smpl, ctx, -1);

            if (token == eos) break;

            std::string piece = token_to_piece(vocab, token);
            auto* data = new StreamData{piece, false, ""};
            tsfn.BlockingCall(data);

            llama_sampler_accept(smpl, token);

            llama_batch next = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, next) != 0) break;
        }

        return_sampler(ctx, smpl, smpl_params, has_grammar);

        // Signal done
        auto* done = new StreamData{"", true, ""};
        tsfn.BlockingCall(done);
        tsfn.Release();
    }).detach();

    return env.Undefined();
}

// ── Batch Inference (Multi-Sequence) ─────────────────────────────────────────

struct BatchStreamData {
    int32_t seq_index;
    std::string token;
    bool done;
    std::string error;
    std::string finish_reason; // "stop" or "length"
};

void InferBatchCallback(Napi::Env env, Napi::Function jsCallback, void* /*context*/, BatchStreamData* data) {
    if (!data->error.empty()) {
        jsCallback.Call({
            Napi::Error::New(env, data->error).Value(),
            env.Null(),
            Napi::Number::New(env, data->seq_index),
            env.Null()
        });
    } else if (data->done) {
        jsCallback.Call({
            env.Null(),
            env.Null(),
            Napi::Number::New(env, data->seq_index),
            Napi::String::New(env, data->finish_reason)
        });
    } else {
        jsCallback.Call({
            env.Null(),
            Napi::String::New(env, data->token),
            Napi::Number::New(env, data->seq_index),
            env.Null()
        });
    }
    delete data;
}

using BatchTSFN = Napi::TypedThreadSafeFunction<void, BatchStreamData, InferBatchCallback>;

Napi::Value InferBatch(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // inferBatch(model, ctx, prompts[], options[], callback)
    if (info.Length() < 5 || !info[0].IsExternal() || !info[1].IsExternal() ||
        !info[2].IsArray() || !info[3].IsArray() || !info[4].IsFunction()) {
        Napi::TypeError::New(env, "inferBatch(model, ctx, prompts[], options[], callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    llama_context* ctx = info[1].As<Napi::External<llama_context>>().Data();
    Napi::Array promptsArr = info[2].As<Napi::Array>();
    Napi::Array optsArr = info[3].As<Napi::Array>();

    uint32_t n_seq = promptsArr.Length();
    if (n_seq == 0) {
        Napi::TypeError::New(env, "inferBatch: prompts array must not be empty")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Parse prompts
    std::vector<std::string> prompts(n_seq);
    for (uint32_t i = 0; i < n_seq; i++) {
        prompts[i] = promptsArr.Get(i).As<Napi::String>().Utf8Value();
    }

    // Parse per-sequence options
    struct SeqOpts {
        int32_t max_tokens = 512;
        float temperature = 0.7f;
        float top_p = 0.9f;
        int32_t top_k = 40;
        float repeat_penalty = 1.1f;
        float frequency_penalty = 0.0f;
        float presence_penalty = 0.0f;
        uint32_t seed = LLAMA_DEFAULT_SEED;
        std::string grammar;
        std::string grammar_root = "root";
    };

    std::vector<SeqOpts> seq_opts(n_seq);
    for (uint32_t i = 0; i < n_seq; i++) {
        if (i < optsArr.Length() && optsArr.Get(i).IsObject()) {
            Napi::Object opts = optsArr.Get(i).As<Napi::Object>();
            parse_gen_options(opts,
                seq_opts[i].max_tokens, seq_opts[i].temperature, seq_opts[i].top_p,
                seq_opts[i].top_k, seq_opts[i].repeat_penalty,
                seq_opts[i].frequency_penalty, seq_opts[i].presence_penalty,
                seq_opts[i].seed, seq_opts[i].grammar, seq_opts[i].grammar_root);
        }
    }

    BatchTSFN tsfn = BatchTSFN::New(env, info[4].As<Napi::Function>(), "InferBatch", 0, 1);

    std::thread([=]() mutable {
        const llama_vocab* vocab = llama_model_get_vocab(model);
        llama_token eos = llama_vocab_eos(vocab);
        int32_t n_batch_size = llama_n_batch(ctx);

        // 1. Tokenize all prompts
        std::vector<std::vector<llama_token>> prompt_tokens(n_seq);
        int32_t total_prompt_tokens = 0;
        for (uint32_t i = 0; i < n_seq; i++) {
            int32_t n = llama_tokenize(vocab, prompts[i].c_str(), prompts[i].size(), nullptr, 0, true, false);
            if (n >= 0) n = 0; else n = -n;
            prompt_tokens[i].resize(n);
            llama_tokenize(vocab, prompts[i].c_str(), prompts[i].size(), prompt_tokens[i].data(), n, true, false);
            total_prompt_tokens += n;
        }

        // 2. Create one sampler per sequence (no caching)
        std::vector<llama_sampler*> samplers(n_seq);
        for (uint32_t i = 0; i < n_seq; i++) {
            samplers[i] = create_sampler(vocab,
                seq_opts[i].top_k, seq_opts[i].top_p, seq_opts[i].temperature,
                seq_opts[i].repeat_penalty, seq_opts[i].frequency_penalty,
                seq_opts[i].presence_penalty, seq_opts[i].seed,
                seq_opts[i].grammar, seq_opts[i].grammar_root, nullptr);
        }

        // 3. Clear KV cache for each sequence
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            for (uint32_t i = 0; i < n_seq; i++) {
                llama_memory_seq_rm(mem, static_cast<llama_seq_id>(i), 0, -1);
            }
        }

        // 4. Prompt eval — build multi-sequence batch with chunking
        {
            // Build full list of (token, pos, seq_id, is_last) entries
            struct PromptEntry {
                llama_token token;
                llama_pos pos;
                llama_seq_id seq_id;
                bool logits; // true only for the last token of each sequence
            };

            std::vector<PromptEntry> entries;
            entries.reserve(total_prompt_tokens);
            for (uint32_t s = 0; s < n_seq; s++) {
                int32_t n_tok = static_cast<int32_t>(prompt_tokens[s].size());
                for (int32_t j = 0; j < n_tok; j++) {
                    entries.push_back({
                        prompt_tokens[s][j],
                        j,
                        static_cast<llama_seq_id>(s),
                        j == n_tok - 1  // logits=1 for last token of sequence
                    });
                }
            }

            // Process in chunks of n_batch_size
            for (size_t offset = 0; offset < entries.size(); offset += n_batch_size) {
                size_t chunk_end = std::min(entries.size(), offset + n_batch_size);
                int32_t chunk_size = static_cast<int32_t>(chunk_end - offset);

                llama_batch batch = llama_batch_init(chunk_size, 0, n_seq);
                for (int32_t i = 0; i < chunk_size; i++) {
                    auto& e = entries[offset + i];
                    batch.token[i] = e.token;
                    batch.pos[i] = e.pos;
                    batch.n_seq_id[i] = 1;
                    batch.seq_id[i][0] = e.seq_id;

                    // Only set logits=1 for last token of each sequence,
                    // but only in the chunk where it actually appears last
                    if (e.logits) {
                        // Verify this is really the last occurrence of this seq_id in remaining entries
                        bool is_last_in_all = true;
                        for (size_t j = offset + i + 1; j < entries.size(); j++) {
                            if (entries[j].seq_id == e.seq_id) {
                                is_last_in_all = false;
                                break;
                            }
                        }
                        batch.logits[i] = is_last_in_all ? 1 : 0;
                    } else {
                        batch.logits[i] = 0;
                    }
                }
                batch.n_tokens = chunk_size;

                int32_t ret = llama_decode(ctx, batch);
                llama_batch_free(batch);

                if (ret != 0) {
                    auto* data = new BatchStreamData{0, "", false, "inferBatch: prompt eval failed", ""};
                    tsfn.BlockingCall(data);
                    for (auto* s : samplers) llama_sampler_free(s);
                    tsfn.Release();
                    return;
                }
            }
        }

        // 5. Generation loop — interleaved multi-sequence decoding
        std::vector<bool> seq_done(n_seq, false);
        std::vector<int32_t> seq_generated(n_seq, 0);
        uint32_t n_active = n_seq;

        // Sample first token for each active sequence
        // After prompt eval, logits indices correspond to sequences with logits=1
        // in batch order. We need to find these indices.
        {
            // The logits output array has entries for each token with logits=1 in the last batch.
            // We need to find the logits index for each sequence's last prompt token.
            // Since we set logits=1 only for the last token of each sequence, the indices
            // are simply 0, 1, 2, ... n_seq-1 in order of appearance.
            int32_t logits_idx = 0;

            // But we need to be more careful: count which sequences had their
            // last token in the final batch chunk
            // For simplicity, since the last chunk contains the final tokens,
            // the logits indices match the order of appearance of seq_ids with logits=1
            // in the last decoded batch.

            // Rebuild which sequences had logits=1 in the last batch
            // Actually, after the chunked prompt eval, the logits in the context
            // correspond to the last decoded batch. We need the index within that batch
            // of tokens that had logits[i]=1.

            // Safer approach: use llama_sampler_sample with the batch index = -1
            // which samples from the last logits. But with multiple sequences,
            // we need the per-sequence logits index.

            // The correct approach: after prompt eval, each sequence has its KV cache
            // filled. We now build a generation batch with one token per sequence.
            // But first we need to sample the first token for each sequence.

            // For the first token sampling, we need to find the logits output index
            // for each sequence. The logits array after decode contains entries only
            // for tokens with batch.logits[i]=1, in batch order.

            // Let's find which tokens in the last decoded batch had logits=1
            // Actually we don't have the last batch anymore. But we know:
            // - entries with logits=true that were in the last chunk

            // Reconstruct: find the logits-enabled entries from the last chunk
            size_t last_chunk_start = 0;
            if (total_prompt_tokens > n_batch_size) {
                size_t full_chunks = total_prompt_tokens / n_batch_size;
                last_chunk_start = full_chunks * n_batch_size;
                if (last_chunk_start >= static_cast<size_t>(total_prompt_tokens))
                    last_chunk_start -= n_batch_size;
            }

            // Count logits entries before each sequence's entry in the last chunk
            // Build a mapping: seq_id -> logits_index in the last batch
            std::vector<int32_t> seq_logits_idx(n_seq, -1);
            int32_t lidx = 0;

            // Re-derive which entries from the last chunk had logits=1
            // We need to reconstruct the entries for the last chunk
            struct PromptEntry2 {
                llama_seq_id seq_id;
                bool logits;
            };
            std::vector<PromptEntry2> last_chunk_entries;
            {
                // Rebuild entries
                std::vector<PromptEntry2> all_entries;
                all_entries.reserve(total_prompt_tokens);
                for (uint32_t s = 0; s < n_seq; s++) {
                    int32_t n_tok = static_cast<int32_t>(prompt_tokens[s].size());
                    for (int32_t j = 0; j < n_tok; j++) {
                        bool is_last = (j == n_tok - 1);
                        // Check if this is truly the last entry for this seq
                        bool is_last_overall = is_last;
                        all_entries.push_back({static_cast<llama_seq_id>(s), is_last_overall});
                    }
                }

                size_t chunk_start = 0;
                size_t chunk_end = 0;
                // Find the last chunk boundaries
                for (size_t off = 0; off < all_entries.size(); off += n_batch_size) {
                    chunk_start = off;
                    chunk_end = std::min(all_entries.size(), off + (size_t)n_batch_size);
                }

                for (size_t i = chunk_start; i < chunk_end; i++) {
                    auto& e = all_entries[i];
                    // Check if this seq_id appears later
                    bool appears_later = false;
                    for (size_t j = i + 1; j < all_entries.size(); j++) {
                        if (all_entries[j].seq_id == e.seq_id) {
                            appears_later = true;
                            break;
                        }
                    }
                    bool has_logits = e.logits && !appears_later;
                    last_chunk_entries.push_back({e.seq_id, has_logits});
                }
            }

            for (auto& e : last_chunk_entries) {
                if (e.logits) {
                    seq_logits_idx[e.seq_id] = lidx;
                    lidx++;
                }
            }

            // Sample first token for each sequence
            for (uint32_t s = 0; s < n_seq; s++) {
                int32_t idx = seq_logits_idx[s];
                if (idx < 0) {
                    // This shouldn't happen — every sequence should have logits
                    auto* data = new BatchStreamData{static_cast<int32_t>(s), "", false,
                        "inferBatch: no logits for sequence " + std::to_string(s), ""};
                    tsfn.BlockingCall(data);
                    seq_done[s] = true;
                    n_active--;
                    continue;
                }

                llama_token token = llama_sampler_sample(samplers[s], ctx, idx);

                if (token == eos) {
                    seq_done[s] = true;
                    n_active--;
                    auto* data = new BatchStreamData{static_cast<int32_t>(s), "", true, "", "stop"};
                    tsfn.BlockingCall(data);
                    continue;
                }

                std::string piece = token_to_piece(vocab, token);
                auto* data = new BatchStreamData{static_cast<int32_t>(s), piece, false, "", ""};
                tsfn.BlockingCall(data);

                llama_sampler_accept(samplers[s], token);
                seq_generated[s]++;

                // We'll need to decode this token — store it for the generation batch
                prompt_tokens[s].push_back(token); // reuse as "last token" storage
            }
        }

        // Generation loop
        while (n_active > 0) {
            // Build generation batch: one token per active sequence
            int32_t batch_count = 0;
            std::vector<uint32_t> active_seqs;
            for (uint32_t s = 0; s < n_seq; s++) {
                if (!seq_done[s]) {
                    active_seqs.push_back(s);
                    batch_count++;
                }
            }

            llama_batch batch = llama_batch_init(batch_count, 0, n_seq);
            for (int32_t i = 0; i < batch_count; i++) {
                uint32_t s = active_seqs[i];
                llama_token last_token = prompt_tokens[s].back();
                int32_t pos = static_cast<int32_t>(prompt_tokens[s].size()) - 1;

                batch.token[i] = last_token;
                batch.pos[i] = pos;
                batch.n_seq_id[i] = 1;
                batch.seq_id[i][0] = static_cast<llama_seq_id>(s);
                batch.logits[i] = 1;
            }
            batch.n_tokens = batch_count;

            int32_t ret = llama_decode(ctx, batch);
            llama_batch_free(batch);

            if (ret != 0) {
                auto* data = new BatchStreamData{0, "", false, "inferBatch: decode failed during generation", ""};
                tsfn.BlockingCall(data);
                break;
            }

            // Sample next token for each active sequence
            for (int32_t i = 0; i < batch_count; i++) {
                uint32_t s = active_seqs[i];
                llama_token token = llama_sampler_sample(samplers[s], ctx, i);

                if (token == eos) {
                    seq_done[s] = true;
                    n_active--;
                    auto* data = new BatchStreamData{static_cast<int32_t>(s), "", true, "", "stop"};
                    tsfn.BlockingCall(data);
                    continue;
                }

                seq_generated[s]++;
                if (seq_generated[s] >= seq_opts[s].max_tokens) {
                    seq_done[s] = true;
                    n_active--;
                    // Send the last token, then signal done with length
                    std::string piece = token_to_piece(vocab, token);
                    auto* tok_data = new BatchStreamData{static_cast<int32_t>(s), piece, false, "", ""};
                    tsfn.BlockingCall(tok_data);
                    auto* done_data = new BatchStreamData{static_cast<int32_t>(s), "", true, "", "length"};
                    tsfn.BlockingCall(done_data);
                    continue;
                }

                std::string piece = token_to_piece(vocab, token);
                auto* data = new BatchStreamData{static_cast<int32_t>(s), piece, false, "", ""};
                tsfn.BlockingCall(data);

                llama_sampler_accept(samplers[s], token);
                prompt_tokens[s].push_back(token);
            }
        }

        // 6. Free all samplers
        for (auto* s : samplers) llama_sampler_free(s);

        // Signal all done
        auto* done = new BatchStreamData{-1, "", true, "", ""};
        tsfn.BlockingCall(done);
        tsfn.Release();
    }).detach();

    return env.Undefined();
}

// ── Log Callback ────────────────────────────────────────────────────────────

struct LogCallbackData {
    std::string text;
    int level;
};

static void LogJsCallback(Napi::Env env, Napi::Function jsCallback, void* /*context*/, LogCallbackData* data) {
    jsCallback.Call({
        Napi::Number::New(env, data->level),
        Napi::String::New(env, data->text)
    });
    delete data;
}

using LogTSFN = Napi::TypedThreadSafeFunction<void, LogCallbackData, LogJsCallback>;

static LogTSFN g_log_tsfn;
static bool g_log_tsfn_active = false;
static int g_log_min_level = 0; // GGML_LOG_LEVEL_NONE = 0 means pass all

static void native_log_callback(enum ggml_log_level level, const char* text, void* /*user_data*/) {
    if (!g_log_tsfn_active) return;
    if (static_cast<int>(level) < g_log_min_level) return;

    auto* data = new LogCallbackData{std::string(text), static_cast<int>(level)};
    g_log_tsfn.BlockingCall(data);
}

Napi::Value SetLogCallback(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // Release previous TSFN if active
    if (g_log_tsfn_active) {
        llama_log_set(nullptr, nullptr);
        g_log_tsfn.Release();
        g_log_tsfn_active = false;
    }

    // If called with null/undefined, just clear the callback
    if (info.Length() < 1 || info[0].IsNull() || info[0].IsUndefined()) {
        return env.Undefined();
    }

    if (!info[0].IsFunction()) {
        Napi::TypeError::New(env, "setLogCallback: argument must be a function or null")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    g_log_tsfn = LogTSFN::New(env, info[0].As<Napi::Function>(), "LogCallback", 0, 1);
    g_log_tsfn_active = true;
    llama_log_set(native_log_callback, nullptr);

    return env.Undefined();
}

Napi::Value SetLogLevel(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "setLogLevel: argument must be a number (0=none, 1=debug, 2=info, 3=warn, 4=error)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    g_log_min_level = info[0].As<Napi::Number>().Int32Value();
    return env.Undefined();
}

// ── Vision / Multimodal (Phase 11) ──────────────────────────────────────────

Napi::Value CreateMtmdContext(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsString()) {
        Napi::TypeError::New(env, "createMtmdContext(model, projectorPath, [options])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    std::string projector_path = info[1].As<Napi::String>().Utf8Value();

    mtmd_context_params params = mtmd_context_params_default();

    if (info.Length() >= 3 && info[2].IsObject()) {
        Napi::Object opts = info[2].As<Napi::Object>();
        if (opts.Has("use_gpu"))
            params.use_gpu = opts.Get("use_gpu").As<Napi::Boolean>().Value();
        if (opts.Has("n_threads"))
            params.n_threads = opts.Get("n_threads").As<Napi::Number>().Int32Value();
    }

    mtmd_context* ctx = mtmd_init_from_file(projector_path.c_str(), model, params);
    if (!ctx) {
        Napi::Error::New(env, "createMtmdContext: failed to load projector from " + projector_path)
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    register_handle(ctx);
    return Napi::External<mtmd_context>::New(env, ctx);
}

Napi::Value FreeMtmdContext(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() >= 1 && info[0].IsExternal()) {
        mtmd_context* ctx = info[0].As<Napi::External<mtmd_context>>().Data();
        if (unregister_handle(ctx)) {
            mtmd_free(ctx);
        }
    }
    return env.Undefined();
}

Napi::Value SupportVision(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "supportVision(mtmdCtx)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    mtmd_context* ctx = info[0].As<Napi::External<mtmd_context>>().Data();
    return Napi::Boolean::New(env, mtmd_support_vision(ctx));
}

// Helper: build bitmaps from JS Buffer array
struct BitmapResult {
    std::vector<mtmd_bitmap*> bitmaps;
    std::string error;
};

static BitmapResult build_bitmaps(mtmd_context* mtmd_ctx, const Napi::Array& bufArr) {
    BitmapResult result;
    uint32_t n = bufArr.Length();
    result.bitmaps.reserve(n);

    for (uint32_t i = 0; i < n; i++) {
        Napi::Buffer<uint8_t> buf = bufArr.Get(i).As<Napi::Buffer<uint8_t>>();
        mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(
            mtmd_ctx, buf.Data(), buf.Length());
        if (!bmp) {
            // Clean up already-created bitmaps
            for (auto* b : result.bitmaps) mtmd_bitmap_free(b);
            result.bitmaps.clear();
            result.error = "Failed to decode image at index " + std::to_string(i);
            return result;
        }
        result.bitmaps.push_back(bmp);
    }
    return result;
}

Napi::Value InferSyncVision(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // inferSyncVision(model, ctx, mtmdCtx, prompt, imageBuffers, [options])
    if (info.Length() < 5 || !info[0].IsExternal() || !info[1].IsExternal() ||
        !info[2].IsExternal() || !info[3].IsString() || !info[4].IsArray()) {
        Napi::TypeError::New(env, "inferSyncVision(model, ctx, mtmdCtx, prompt, imageBuffers, [options])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    llama_context* ctx = info[1].As<Napi::External<llama_context>>().Data();
    mtmd_context* mtmd_ctx = info[2].As<Napi::External<mtmd_context>>().Data();
    std::string prompt = info[3].As<Napi::String>().Utf8Value();
    Napi::Array imgArr = info[4].As<Napi::Array>();

    int32_t max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    float repeat_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    std::string grammar_str;
    std::string grammar_root = "root";

    if (info.Length() >= 6 && info[5].IsObject()) {
        parse_gen_options(info[5].As<Napi::Object>(),
            max_tokens, temperature, top_p, top_k, repeat_penalty,
            frequency_penalty, presence_penalty, seed,
            grammar_str, grammar_root);
    }

    // Build bitmaps from image buffers
    BitmapResult bmpResult = build_bitmaps(mtmd_ctx, imgArr);
    if (!bmpResult.error.empty()) {
        Napi::Error::New(env, "inferSyncVision: " + bmpResult.error)
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Tokenize: split prompt into text + image chunks
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text = { prompt.c_str(), true, false };

    std::vector<const mtmd_bitmap*> bmp_ptrs(bmpResult.bitmaps.size());
    for (size_t i = 0; i < bmpResult.bitmaps.size(); i++) {
        bmp_ptrs[i] = bmpResult.bitmaps[i];
    }

    int32_t tok_res = mtmd_tokenize(mtmd_ctx, chunks, &input_text,
        bmp_ptrs.data(), bmp_ptrs.size());

    // Free bitmaps — no longer needed after tokenization
    for (auto* b : bmpResult.bitmaps) mtmd_bitmap_free(b);

    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        std::string msg = tok_res == 1
            ? "number of images doesn't match number of markers in prompt"
            : "image preprocessing failed";
        Napi::Error::New(env, "inferSyncVision: " + msg)
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Evaluate chunks (text + image embeddings)
    llama_pos n_past = 0;
    int32_t n_batch = llama_n_batch(ctx);
    int32_t eval_res = mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, n_past, 0,
        n_batch, true, &n_past);

    mtmd_input_chunks_free(chunks);

    if (eval_res != 0) {
        Napi::Error::New(env, "inferSyncVision: chunk evaluation failed")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Sampling loop (same as InferSync)
    const llama_vocab* vocab = llama_model_get_vocab(model);
    SamplerParams smpl_params{top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed};
    llama_sampler* smpl = create_sampler(vocab, top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed, grammar_str, grammar_root, ctx);
    bool has_grammar = !grammar_str.empty();
    llama_token eos = llama_vocab_eos(vocab);
    std::string result;

    for (int32_t i = 0; i < max_tokens; i++) {
        llama_token token = llama_sampler_sample(smpl, ctx, -1);
        if (token == eos) break;

        token_append(vocab, token, result);
        llama_sampler_accept(smpl, token);

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, next) != 0) break;
    }

    return_sampler(ctx, smpl, smpl_params, has_grammar);
    return Napi::String::New(env, result);
}

Napi::Value InferStreamVision(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    // inferStreamVision(model, ctx, mtmdCtx, prompt, imageBuffers, options, callback)
    if (info.Length() < 7 || !info[0].IsExternal() || !info[1].IsExternal() ||
        !info[2].IsExternal() || !info[3].IsString() || !info[4].IsArray() ||
        !info[6].IsFunction()) {
        Napi::TypeError::New(env, "inferStreamVision(model, ctx, mtmdCtx, prompt, imageBuffers, options, callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    llama_model* model = info[0].As<Napi::External<llama_model>>().Data();
    llama_context* ctx = info[1].As<Napi::External<llama_context>>().Data();
    mtmd_context* mtmd_ctx = info[2].As<Napi::External<mtmd_context>>().Data();
    std::string prompt = info[3].As<Napi::String>().Utf8Value();
    Napi::Array imgArr = info[4].As<Napi::Array>();

    int32_t max_tokens = 512;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int32_t top_k = 40;
    float repeat_penalty = 1.1f;
    float frequency_penalty = 0.0f;
    float presence_penalty = 0.0f;
    uint32_t seed = LLAMA_DEFAULT_SEED;
    std::string grammar_str;
    std::string grammar_root = "root";

    if (info[5].IsObject()) {
        parse_gen_options(info[5].As<Napi::Object>(),
            max_tokens, temperature, top_p, top_k, repeat_penalty,
            frequency_penalty, presence_penalty, seed,
            grammar_str, grammar_root);
    }

    // Build bitmaps on the main thread (need JS Buffer access)
    BitmapResult bmpResult = build_bitmaps(mtmd_ctx, imgArr);
    if (!bmpResult.error.empty()) {
        Napi::Error::New(env, "inferStreamVision: " + bmpResult.error)
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Tokenize on the main thread (needs mtmd_ctx)
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    mtmd_input_text input_text = { prompt.c_str(), true, false };

    std::vector<const mtmd_bitmap*> bmp_ptrs(bmpResult.bitmaps.size());
    for (size_t i = 0; i < bmpResult.bitmaps.size(); i++) {
        bmp_ptrs[i] = bmpResult.bitmaps[i];
    }

    int32_t tok_res = mtmd_tokenize(mtmd_ctx, chunks, &input_text,
        bmp_ptrs.data(), bmp_ptrs.size());

    for (auto* b : bmpResult.bitmaps) mtmd_bitmap_free(b);

    if (tok_res != 0) {
        mtmd_input_chunks_free(chunks);
        std::string msg = tok_res == 1
            ? "number of images doesn't match number of markers in prompt"
            : "image preprocessing failed";
        Napi::Error::New(env, "inferStreamVision: " + msg)
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    TSFN tsfn = TSFN::New(env, info[6].As<Napi::Function>(), "InferStreamVision", 0, 1);

    std::thread([=]() mutable {
        // Evaluate chunks
        llama_pos n_past = 0;
        int32_t n_batch = llama_n_batch(ctx);
        int32_t eval_res = mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, n_past, 0,
            n_batch, true, &n_past);

        mtmd_input_chunks_free(chunks);

        if (eval_res != 0) {
            auto* data = new StreamData{"", false, "inferStreamVision: chunk evaluation failed"};
            tsfn.BlockingCall(data);
            tsfn.Release();
            return;
        }

        // Sampling loop
        const llama_vocab* vocab = llama_model_get_vocab(model);
        SamplerParams smpl_params{top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed};
        llama_sampler* smpl = create_sampler(vocab, top_k, top_p, temperature, repeat_penalty, frequency_penalty, presence_penalty, seed, grammar_str, grammar_root, ctx);
        bool has_grammar = !grammar_str.empty();
        llama_token eos = llama_vocab_eos(vocab);

        for (int32_t i = 0; i < max_tokens; i++) {
            llama_token token = llama_sampler_sample(smpl, ctx, -1);
            if (token == eos) break;

            std::string piece = token_to_piece(vocab, token);
            auto* data = new StreamData{piece, false, ""};
            tsfn.BlockingCall(data);

            llama_sampler_accept(smpl, token);
            llama_batch next = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, next) != 0) break;
        }

        return_sampler(ctx, smpl, smpl_params, has_grammar);

        auto* done = new StreamData{"", true, ""};
        tsfn.BlockingCall(done);
        tsfn.Release();
    }).detach();

    return env.Undefined();
}

// ── JSON Schema to Grammar ──────────────────────────────────────────────────

Napi::Value JsonSchemaToGrammar(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "jsonSchemaToGrammar(schemaJson: string)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string schema_json = info[0].As<Napi::String>().Utf8Value();

    try {
        auto schema = nlohmann::ordered_json::parse(schema_json);
        std::string grammar = json_schema_to_grammar(schema);
        return Napi::String::New(env, grammar);
    } catch (const std::exception& e) {
        Napi::Error::New(env, std::string("jsonSchemaToGrammar: ") + e.what())
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
}

// ── Quantize ────────────────────────────────────────────────────────────────

struct QuantizeCallbackData {
    std::string error; // empty on success
};

void QuantizeJsCallback(Napi::Env env, Napi::Function jsCallback, void* /*context*/, QuantizeCallbackData* data) {
    if (data->error.empty()) {
        jsCallback.Call({env.Null()});
    } else {
        jsCallback.Call({Napi::Error::New(env, data->error).Value()});
    }
    delete data;
}

using QuantizeTSFN = Napi::TypedThreadSafeFunction<void, QuantizeCallbackData, QuantizeJsCallback>;

static int string_to_ftype(const std::string& s) {
    if (s == "F32")      return 0;
    if (s == "F16")      return 1;
    if (s == "Q4_0")     return 2;
    if (s == "Q4_1")     return 3;
    if (s == "Q8_0")     return 7;
    if (s == "Q5_0")     return 8;
    if (s == "Q5_1")     return 9;
    if (s == "Q2_K")     return 10;
    if (s == "Q3_K_S")   return 11;
    if (s == "Q3_K_M")   return 12;
    if (s == "Q3_K_L")   return 13;
    if (s == "Q4_K_S")   return 14;
    if (s == "Q4_K_M")   return 15;
    if (s == "Q5_K_S")   return 16;
    if (s == "Q5_K_M")   return 17;
    if (s == "Q6_K")     return 18;
    if (s == "IQ2_XXS")  return 19;
    if (s == "IQ2_XS")   return 20;
    if (s == "IQ3_XS")   return 22;
    if (s == "IQ3_XXS")  return 23;
    if (s == "IQ1_S")    return 24;
    if (s == "IQ4_NL")   return 25;
    if (s == "IQ3_S")    return 26;
    if (s == "IQ3_M")    return 27;
    if (s == "IQ2_S")    return 28;
    if (s == "IQ2_M")    return 29;
    if (s == "IQ4_XS")   return 30;
    if (s == "IQ1_M")    return 31;
    if (s == "BF16")     return 32;
    if (s == "TQ1_0")    return 36;
    if (s == "TQ2_0")    return 37;
    return -1;
}

Napi::Value Quantize(const Napi::CallbackInfo& info) {
    Napi::Env env = info.Env();

    if (info.Length() < 4 || !info[0].IsString() || !info[1].IsString() ||
        !info[2].IsObject() || !info[3].IsFunction()) {
        Napi::TypeError::New(env, "quantize(inputPath, outputPath, options, callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    ensure_backend();

    std::string input_path = info[0].As<Napi::String>().Utf8Value();
    std::string output_path = info[1].As<Napi::String>().Utf8Value();
    Napi::Object opts = info[2].As<Napi::Object>();

    // Parse ftype
    int ftype = 15; // Q4_K_M default
    if (opts.Has("ftype")) {
        ftype = opts.Get("ftype").As<Napi::Number>().Int32Value();
    }

    // Parse optional params
    int nthread = 0; // 0 = llama.cpp auto-detects
    bool allow_requantize = false;
    bool quantize_output_tensor = true;
    bool pure = false;

    if (opts.Has("nthread"))                nthread = opts.Get("nthread").As<Napi::Number>().Int32Value();
    if (opts.Has("allow_requantize"))       allow_requantize = opts.Get("allow_requantize").As<Napi::Boolean>().Value();
    if (opts.Has("quantize_output_tensor")) quantize_output_tensor = opts.Get("quantize_output_tensor").As<Napi::Boolean>().Value();
    if (opts.Has("pure"))                   pure = opts.Get("pure").As<Napi::Boolean>().Value();

    QuantizeTSFN tsfn = QuantizeTSFN::New(env, info[3].As<Napi::Function>(), "Quantize", 0, 1);

    std::thread([=]() {
        llama_model_quantize_params params = llama_model_quantize_default_params();
        params.ftype = (llama_ftype)ftype;
        params.nthread = nthread;
        params.allow_requantize = allow_requantize;
        params.quantize_output_tensor = quantize_output_tensor;
        params.pure = pure;

        int result = llama_model_quantize(input_path.c_str(), output_path.c_str(), &params);

        auto* data = new QuantizeCallbackData{};
        if (result != 0) {
            data->error = "Quantization failed (llama_model_quantize returned " + std::to_string(result) + ")";
        }
        tsfn.BlockingCall(data);
        tsfn.Release();
    }).detach();

    return env.Undefined();
}

// ── Module Init ─────────────────────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("backendInfo",        Napi::Function::New(env, BackendInfo));
    exports.Set("backendVersion",     Napi::Function::New(env, BackendVersion));
    exports.Set("loadModel",          Napi::Function::New(env, LoadModel));
    exports.Set("loadModelFromBuffer", Napi::Function::New(env, LoadModelFromBuffer));
    exports.Set("getModelSize",       Napi::Function::New(env, GetModelSize));
    exports.Set("createContext",      Napi::Function::New(env, CreateContext));
    exports.Set("freeModel",          Napi::Function::New(env, FreeModel));
    exports.Set("freeContext",        Napi::Function::New(env, FreeContext));
    exports.Set("getContextSize",    Napi::Function::New(env, GetContextSize));
    exports.Set("tokenize",           Napi::Function::New(env, Tokenize));
    exports.Set("detokenize",         Napi::Function::New(env, Detokenize));
    exports.Set("applyChatTemplate",  Napi::Function::New(env, ApplyChatTemplate));
    exports.Set("inferSync",          Napi::Function::New(env, InferSync));
    exports.Set("inferStream",        Napi::Function::New(env, InferStream));
    exports.Set("kvCacheClear",       Napi::Function::New(env, KvCacheClear));
    exports.Set("setLogCallback",     Napi::Function::New(env, SetLogCallback));
    exports.Set("setLogLevel",        Napi::Function::New(env, SetLogLevel));
    exports.Set("createMtmdContext",  Napi::Function::New(env, CreateMtmdContext));
    exports.Set("freeMtmdContext",    Napi::Function::New(env, FreeMtmdContext));
    exports.Set("supportVision",      Napi::Function::New(env, SupportVision));
    exports.Set("inferSyncVision",    Napi::Function::New(env, InferSyncVision));
    exports.Set("inferStreamVision",  Napi::Function::New(env, InferStreamVision));
    exports.Set("jsonSchemaToGrammar", Napi::Function::New(env, JsonSchemaToGrammar));
    exports.Set("getEmbeddingDimension", Napi::Function::New(env, GetEmbeddingDimension));
    exports.Set("createEmbeddingContext", Napi::Function::New(env, CreateEmbeddingContext));
    exports.Set("embed",                 Napi::Function::New(env, Embed));
    exports.Set("embedBatch",            Napi::Function::New(env, EmbedBatch));
    exports.Set("quantize",              Napi::Function::New(env, Quantize));
    exports.Set("inferBatch",            Napi::Function::New(env, InferBatch));
    return exports;
}

NODE_API_MODULE(hilum_native, Init)
