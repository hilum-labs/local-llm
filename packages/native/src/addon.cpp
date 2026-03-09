/*
 * addon.cpp — N-API bridge for local-llm (Node.js / Bun / Electron).
 *
 * This is a thin wrapper that converts between JS values and the
 * platform-neutral libhilum C API. All inference logic lives in
 * hilum_llm.cpp inside the engine.
 */

#include <napi.h>
#include <thread>
#include <string>
#include <vector>
#include <cstring>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#ifdef __linux__
#include <fstream>
#endif

#include "hilum_llm.h"
#include <nlohmann/json.hpp>

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::string js_string(const Napi::Value & v) {
    return v.As<Napi::String>().Utf8Value();
}

static hilum_model * get_model(const Napi::Value & v) {
    return v.As<Napi::External<hilum_model>>().Data();
}

static hilum_context * get_ctx(const Napi::Value & v) {
    return v.As<Napi::External<hilum_context>>().Data();
}

// ── Backend info ────────────────────────────────────────────────────────────

Napi::Value BackendInfo(const Napi::CallbackInfo & info) {
    return Napi::String::New(info.Env(), hilum_backend_info());
}

Napi::Value BackendVersion(const Napi::CallbackInfo & info) {
    return Napi::String::New(info.Env(), hilum_backend_version());
}

// ── Model lifecycle ─────────────────────────────────────────────────────────

Napi::Value LoadModel(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "loadModel: first argument must be a string path")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model_params params = hilum_model_default_params();
    std::string path = js_string(info[0]);

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("n_gpu_layers"))
            params.n_gpu_layers = opts.Get("n_gpu_layers").As<Napi::Number>().Int32Value();
        if (opts.Has("use_mmap"))
            params.use_mmap = opts.Get("use_mmap").As<Napi::Boolean>().Value();
    }

    hilum_model * model = nullptr;
    hilum_error err = hilum_model_load(path.c_str(), params, &model);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("loadModel: ") + hilum_error_str(err) + " from " + path)
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return Napi::External<hilum_model>::New(env, model);
}

Napi::Value LoadModelFromBuffer(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsBuffer()) {
        Napi::TypeError::New(env, "First argument must be a Buffer")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Buffer<uint8_t> buf = info[0].As<Napi::Buffer<uint8_t>>();
    hilum_model_params params = hilum_model_default_params();

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("n_gpu_layers"))
            params.n_gpu_layers = opts.Get("n_gpu_layers").As<Napi::Number>().Int32Value();
        if (opts.Has("use_mmap"))
            params.use_mmap = opts.Get("use_mmap").As<Napi::Boolean>().Value();
    }

    hilum_model * model = nullptr;
    hilum_error err = hilum_model_load_from_buffer(buf.Data(), buf.Length(), params, &model);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("loadModelFromBuffer: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    // Hold persistent ref to JS Buffer so GC won't collect it while model is alive
    auto * ref = new Napi::Reference<Napi::Buffer<uint8_t>>(Napi::Persistent(buf));
    return Napi::External<hilum_model>::New(env, model,
        [ref](Napi::Env, hilum_model * m) { hilum_model_free(m); delete ref; });
}

Napi::Value GetModelSize(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "getModelSize: first argument must be a model external")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Number::New(env, static_cast<double>(hilum_model_size(get_model(info[0]))));
}

Napi::Value FreeModel(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() >= 1 && info[0].IsExternal())
        hilum_model_free(get_model(info[0]));
    return env.Undefined();
}

// ── Context lifecycle ───────────────────────────────────────────────────────

Napi::Value CreateContext(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "createContext: first argument must be a model external")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_context_params params = hilum_context_default_params();
    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("n_ctx"))           params.n_ctx = opts.Get("n_ctx").As<Napi::Number>().Uint32Value();
        if (opts.Has("n_batch"))         params.n_batch = opts.Get("n_batch").As<Napi::Number>().Uint32Value();
        if (opts.Has("n_threads"))       params.n_threads = opts.Get("n_threads").As<Napi::Number>().Uint32Value();
        if (opts.Has("flash_attn_type")) params.flash_attn = opts.Get("flash_attn_type").As<Napi::Number>().Int32Value();
        if (opts.Has("type_k"))          params.type_k = opts.Get("type_k").As<Napi::Number>().Int32Value();
        if (opts.Has("type_v"))          params.type_v = opts.Get("type_v").As<Napi::Number>().Int32Value();
        if (opts.Has("n_seq_max"))       params.n_seq_max = opts.Get("n_seq_max").As<Napi::Number>().Uint32Value();
        if (opts.Has("draft_model") && opts.Get("draft_model").IsExternal())
            params.draft_model = opts.Get("draft_model").As<Napi::External<hilum_model>>().Data();
        if (opts.Has("draft_n_max"))
            params.draft_n_max = opts.Get("draft_n_max").As<Napi::Number>().Int32Value();
    }

    hilum_context * ctx = nullptr;
    hilum_error err = hilum_context_create(get_model(info[0]), params, &ctx);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("createContext: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return Napi::External<hilum_context>::New(env, ctx);
}

Napi::Value FreeContext(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() >= 1 && info[0].IsExternal())
        hilum_context_free(get_ctx(info[0]));
    return env.Undefined();
}

Napi::Value GetContextSize(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "getContextSize(ctx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Number::New(env, static_cast<double>(hilum_context_size(get_ctx(info[0]))));
}

Napi::Value KvCacheClear(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsNumber()) {
        Napi::TypeError::New(env, "kvCacheClear(ctx, fromPos)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    hilum_context_kv_clear(get_ctx(info[0]), info[1].As<Napi::Number>().Int32Value());
    return env.Undefined();
}

// ── Tokenization ────────────────────────────────────────────────────────────

Napi::Value Tokenize(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsString()) {
        Napi::TypeError::New(env, "tokenize(model, text, [addSpecial], [parseSpecial])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);
    std::string text = js_string(info[1]);
    bool add_special  = (info.Length() >= 3 && info[2].IsBoolean()) ? info[2].As<Napi::Boolean>().Value() : true;
    bool parse_special = (info.Length() >= 4 && info[3].IsBoolean()) ? info[3].As<Napi::Boolean>().Value() : false;

    int32_t n = hilum_tokenize(model, text.c_str(), static_cast<int32_t>(text.size()),
                                nullptr, 0, add_special, parse_special);
    if (n >= 0) return Napi::Int32Array::New(env, 0);

    int32_t n_tokens = -n;
    std::vector<int32_t> tokens(n_tokens);
    int32_t result = hilum_tokenize(model, text.c_str(), static_cast<int32_t>(text.size()),
                                     tokens.data(), n_tokens, add_special, parse_special);
    if (result < 0) {
        Napi::Error::New(env, "tokenize: failed").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Int32Array arr = Napi::Int32Array::New(env, result);
    for (int32_t i = 0; i < result; i++) arr[i] = tokens[i];
    return arr;
}

Napi::Value Detokenize(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "detokenize(model, tokens)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);

    std::vector<int32_t> tokens;
    if (info[1].IsTypedArray()) {
        Napi::Int32Array arr = info[1].As<Napi::Int32Array>();
        tokens.resize(arr.ElementLength());
        for (size_t i = 0; i < arr.ElementLength(); i++) tokens[i] = arr[i];
    } else if (info[1].IsArray()) {
        Napi::Array arr = info[1].As<Napi::Array>();
        tokens.resize(arr.Length());
        for (uint32_t i = 0; i < arr.Length(); i++)
            tokens[i] = arr.Get(i).As<Napi::Number>().Int32Value();
    } else {
        Napi::TypeError::New(env, "detokenize: tokens must be Int32Array or Array")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::vector<char> buf(tokens.size() * 16 + 256);
    int32_t n = hilum_detokenize(model, tokens.data(), static_cast<int32_t>(tokens.size()),
                                  buf.data(), static_cast<int32_t>(buf.size()));
    if (n < 0) {
        buf.resize(-n);
        n = hilum_detokenize(model, tokens.data(), static_cast<int32_t>(tokens.size()),
                              buf.data(), static_cast<int32_t>(buf.size()));
    }

    return Napi::String::New(env, buf.data(), n > 0 ? n : 0);
}

// ── Chat Template ───────────────────────────────────────────────────────────

Napi::Value ApplyChatTemplate(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsArray()) {
        Napi::TypeError::New(env, "applyChatTemplate(model, messages, [addAssistant])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);
    Napi::Array msgs_arr = info[1].As<Napi::Array>();
    bool add_assistant = (info.Length() >= 3 && info[2].IsBoolean()) ? info[2].As<Napi::Boolean>().Value() : true;

    // Build JSON array of messages
    nlohmann::json j = nlohmann::json::array();
    for (uint32_t i = 0; i < msgs_arr.Length(); i++) {
        Napi::Object msg = msgs_arr.Get(i).As<Napi::Object>();
        j.push_back({
            {"role",    msg.Get("role").As<Napi::String>().Utf8Value()},
            {"content", msg.Get("content").As<Napi::String>().Utf8Value()}
        });
    }
    std::string messages_json = j.dump();

    std::vector<char> buf(messages_json.size() * 4 + 256);
    int32_t len = hilum_chat_template(model, messages_json.c_str(), add_assistant,
                                       buf.data(), static_cast<int32_t>(buf.size()));
    if (len <= 0) {
        if (len < 0) {
            buf.resize(-len + 1);
            len = hilum_chat_template(model, messages_json.c_str(), add_assistant,
                                       buf.data(), static_cast<int32_t>(buf.size()));
        }
        if (len <= 0) {
            Napi::Error::New(env, "applyChatTemplate: failed").ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    return Napi::String::New(env, buf.data(), len);
}

// ── Generation params helper ────────────────────────────────────────────────

static hilum_gen_params parse_gen_params(const Napi::Object & opts) {
    hilum_gen_params p = hilum_gen_default_params();
    if (opts.Has("max_tokens"))         p.max_tokens = opts.Get("max_tokens").As<Napi::Number>().Int32Value();
    if (opts.Has("temperature"))        p.temperature = opts.Get("temperature").As<Napi::Number>().FloatValue();
    if (opts.Has("top_p"))              p.top_p = opts.Get("top_p").As<Napi::Number>().FloatValue();
    if (opts.Has("top_k"))              p.top_k = opts.Get("top_k").As<Napi::Number>().Int32Value();
    if (opts.Has("repeat_penalty"))     p.repeat_penalty = opts.Get("repeat_penalty").As<Napi::Number>().FloatValue();
    if (opts.Has("frequency_penalty"))  p.frequency_penalty = opts.Get("frequency_penalty").As<Napi::Number>().FloatValue();
    if (opts.Has("presence_penalty"))   p.presence_penalty = opts.Get("presence_penalty").As<Napi::Number>().FloatValue();
    if (opts.Has("seed"))               p.seed = opts.Get("seed").As<Napi::Number>().Uint32Value();
    if (opts.Has("n_past"))             p.n_past = opts.Get("n_past").As<Napi::Number>().Int32Value();
    return p;
}

// Persistent grammar/grammar_root strings (must outlive gen_params usage)
struct GenContext {
    hilum_gen_params params;
    std::string grammar;
    std::string grammar_root;

    void finalize() {
        params.grammar      = grammar.empty()      ? nullptr : grammar.c_str();
        params.grammar_root  = grammar_root.empty() ? nullptr : grammar_root.c_str();
    }
};

static GenContext parse_gen_context(const Napi::Object & opts) {
    GenContext gc;
    gc.params = parse_gen_params(opts);
    if (opts.Has("grammar"))      gc.grammar = opts.Get("grammar").As<Napi::String>().Utf8Value();
    if (opts.Has("grammar_root")) gc.grammar_root = opts.Get("grammar_root").As<Napi::String>().Utf8Value();
    gc.finalize();
    return gc;
}

// ── Prompt-eval progress (synchronous path only) ────────────────────────────

struct SyncProgressCtx {
    Napi::Env env;
    Napi::Function fn;
};

static bool sync_progress_callback(int32_t processed, int32_t total, void * ud) {
    auto * pc = static_cast<SyncProgressCtx *>(ud);
    Napi::Value result = pc->fn.Call({
        Napi::Number::New(pc->env, processed),
        Napi::Number::New(pc->env, total)
    });
    if (result.IsBoolean()) return result.As<Napi::Boolean>().Value();
    return true;
}

// ── Inference — Synchronous ─────────────────────────────────────────────────

Napi::Value InferSync(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsExternal() || !info[1].IsExternal() || !info[2].IsString()) {
        Napi::TypeError::New(env, "inferSync(model, ctx, prompt, [options])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);
    hilum_context * ctx = get_ctx(info[1]);
    std::string prompt = js_string(info[2]);

    GenContext gc;
    gc.params = hilum_gen_default_params();
    if (info.Length() >= 4 && info[3].IsObject())
        gc = parse_gen_context(info[3].As<Napi::Object>());
    else
        gc.finalize();

    SyncProgressCtx * progress_ctx = nullptr;
    if (info.Length() >= 4 && info[3].IsObject()) {
        Napi::Object opts = info[3].As<Napi::Object>();
        if (opts.Has("onPromptProgress") && opts.Get("onPromptProgress").IsFunction()) {
            progress_ctx = new SyncProgressCtx{env, opts.Get("onPromptProgress").As<Napi::Function>()};
            gc.params.progress_callback  = sync_progress_callback;
            gc.params.progress_user_data = progress_ctx;
        }
    }

    std::vector<char> buf(gc.params.max_tokens * 64 + 1024);
    int32_t generated = 0;

    hilum_error err = hilum_generate(model, ctx, prompt.c_str(), gc.params,
                                      buf.data(), static_cast<int32_t>(buf.size()), &generated);
    delete progress_ctx;

    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("inferSync: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return Napi::String::New(env, buf.data());
}

// ── Inference — Streaming (Async) ───────────────────────────────────────────

struct StreamData {
    std::string token;
    bool done;
    std::string error;
};

void InferStreamCallback(Napi::Env env, Napi::Function jsCallback, void *, StreamData * data) {
    if (!data->error.empty()) {
        jsCallback.Call({Napi::Error::New(env, data->error).Value(), env.Null()});
    } else if (data->done) {
        jsCallback.Call({env.Null(), env.Null()});
    } else {
        jsCallback.Call({env.Null(), Napi::String::New(env, data->token)});
    }
    delete data;
}

using TSFN = Napi::TypedThreadSafeFunction<void, StreamData, InferStreamCallback>;

Napi::Value InferStream(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5 || !info[0].IsExternal() || !info[1].IsExternal() ||
        !info[2].IsString() || !info[4].IsFunction()) {
        Napi::TypeError::New(env, "inferStream(model, ctx, prompt, options, callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);
    hilum_context * ctx = get_ctx(info[1]);
    std::string prompt = js_string(info[2]);

    GenContext gc;
    gc.params = hilum_gen_default_params();
    if (info[3].IsObject())
        gc = parse_gen_context(info[3].As<Napi::Object>());
    else
        gc.finalize();

    TSFN tsfn = TSFN::New(env, info[4].As<Napi::Function>(), "InferStream", 0, 1);

    std::thread([model, ctx, prompt, gc, tsfn]() mutable {
        hilum_error err = hilum_generate_stream(model, ctx, prompt.c_str(), gc.params,
            [](const char * token, int32_t token_len, void * ud) -> bool {
                auto * tsfn_ptr = static_cast<TSFN *>(ud);
                auto * data = new StreamData{std::string(token, token_len), false, ""};
                tsfn_ptr->BlockingCall(data);
                return true;
            }, &tsfn);

        if (err != HILUM_OK && err != HILUM_ERR_CANCELLED) {
            auto * data = new StreamData{"", false, hilum_error_str(err)};
            tsfn.BlockingCall(data);
        }

        auto * done = new StreamData{"", true, ""};
        tsfn.BlockingCall(done);
        tsfn.Release();
    }).detach();

    return env.Undefined();
}

// ── Embeddings ───────────────────────────────────────────────────────────────

Napi::Value GetEmbeddingDimension(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "getEmbeddingDimension(model)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Number::New(env, hilum_emb_dimension(get_model(info[0])));
}

Napi::Value CreateEmbeddingContext(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "createEmbeddingContext(model, opts?)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_emb_params params;
    params.n_ctx        = 0;
    params.n_batch      = 0;
    params.n_threads    = 0;
    params.pooling_type = -1;

    if (info.Length() >= 2 && info[1].IsObject()) {
        Napi::Object opts = info[1].As<Napi::Object>();
        if (opts.Has("n_ctx"))         params.n_ctx = opts.Get("n_ctx").As<Napi::Number>().Uint32Value();
        if (opts.Has("n_batch"))       params.n_batch = opts.Get("n_batch").As<Napi::Number>().Uint32Value();
        if (opts.Has("n_threads"))     params.n_threads = opts.Get("n_threads").As<Napi::Number>().Uint32Value();
        if (opts.Has("pooling_type"))  params.pooling_type = opts.Get("pooling_type").As<Napi::Number>().Int32Value();
    }

    hilum_emb_ctx * ec = nullptr;
    hilum_error err = hilum_emb_context_create(get_model(info[0]), params, &ec);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("createEmbeddingContext: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return Napi::External<hilum_emb_ctx>::New(env, ec);
}

Napi::Value Embed(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsExternal() || !info[1].IsExternal() || !info[2].IsTypedArray()) {
        Napi::TypeError::New(env, "embed(ctx, model, Int32Array tokens)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_emb_ctx * ctx = info[0].As<Napi::External<hilum_emb_ctx>>().Data();
    hilum_model * model = get_model(info[1]);
    Napi::Int32Array tokens = info[2].As<Napi::Int32Array>();

    int32_t n_tokens = static_cast<int32_t>(tokens.ElementLength());
    int32_t n_embd = hilum_emb_dimension(model);
    if (n_tokens == 0) return Napi::Float32Array::New(env, 0);

    std::vector<int32_t> tok_vec(n_tokens);
    for (int32_t i = 0; i < n_tokens; i++) tok_vec[i] = tokens[i];

    Napi::Float32Array result = Napi::Float32Array::New(env, n_embd);
    std::vector<float> emb(n_embd);

    hilum_error err = hilum_embed(ctx, model, tok_vec.data(), n_tokens, emb.data(), n_embd);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("embed: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    for (int32_t i = 0; i < n_embd; i++) result[i] = emb[i];
    return result;
}

Napi::Value EmbedBatch(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 3 || !info[0].IsExternal() || !info[1].IsExternal() || !info[2].IsArray()) {
        Napi::TypeError::New(env, "embedBatch(ctx, model, Array<Int32Array> tokenArrays)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_emb_ctx * ctx = info[0].As<Napi::External<hilum_emb_ctx>>().Data();
    hilum_model * model = get_model(info[1]);
    Napi::Array tokenArrays = info[2].As<Napi::Array>();

    int32_t n_embd = hilum_emb_dimension(model);
    uint32_t n_inputs = tokenArrays.Length();
    if (n_inputs == 0) return Napi::Array::New(env, 0);

    // Build token arrays
    std::vector<std::vector<int32_t>> token_vecs(n_inputs);
    std::vector<const int32_t *> token_ptrs(n_inputs);
    std::vector<int32_t> token_counts(n_inputs);

    for (uint32_t i = 0; i < n_inputs; i++) {
        Napi::Int32Array arr = tokenArrays.Get(i).As<Napi::Int32Array>();
        token_vecs[i].resize(arr.ElementLength());
        for (size_t j = 0; j < arr.ElementLength(); j++) token_vecs[i][j] = arr[j];
        token_ptrs[i] = token_vecs[i].data();
        token_counts[i] = static_cast<int32_t>(token_vecs[i].size());
    }

    // Allocate output
    std::vector<std::vector<float>> emb_vecs(n_inputs, std::vector<float>(n_embd));
    std::vector<float *> emb_ptrs(n_inputs);
    for (uint32_t i = 0; i < n_inputs; i++) emb_ptrs[i] = emb_vecs[i].data();

    hilum_error err = hilum_embed_batch(ctx, model, token_ptrs.data(), token_counts.data(),
                                         static_cast<int32_t>(n_inputs), emb_ptrs.data(), n_embd);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("embedBatch: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Array results = Napi::Array::New(env, n_inputs);
    for (uint32_t i = 0; i < n_inputs; i++) {
        Napi::Float32Array vec = Napi::Float32Array::New(env, n_embd);
        for (int32_t j = 0; j < n_embd; j++) vec[j] = emb_vecs[i][j];
        results.Set(i, vec);
    }
    return results;
}

// ── Batch Inference ──────────────────────────────────────────────────────────

struct BatchStreamData {
    int32_t     seq_index;
    std::string token;
    bool        done;
    std::string error;
    std::string finish_reason;
};

void InferBatchCallback(Napi::Env env, Napi::Function jsCallback, void *, BatchStreamData * data) {
    if (!data->error.empty()) {
        jsCallback.Call({
            Napi::Error::New(env, data->error).Value(), env.Null(),
            Napi::Number::New(env, data->seq_index), env.Null()
        });
    } else if (data->done) {
        jsCallback.Call({
            env.Null(), env.Null(),
            Napi::Number::New(env, data->seq_index),
            Napi::String::New(env, data->finish_reason)
        });
    } else {
        jsCallback.Call({
            env.Null(), Napi::String::New(env, data->token),
            Napi::Number::New(env, data->seq_index), env.Null()
        });
    }
    delete data;
}

using BatchTSFN = Napi::TypedThreadSafeFunction<void, BatchStreamData, InferBatchCallback>;

Napi::Value InferBatch(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5 || !info[0].IsExternal() || !info[1].IsExternal() ||
        !info[2].IsArray() || !info[3].IsArray() || !info[4].IsFunction()) {
        Napi::TypeError::New(env, "inferBatch(model, ctx, prompts[], options[], callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);
    hilum_context * ctx = get_ctx(info[1]);
    Napi::Array promptsArr = info[2].As<Napi::Array>();

    uint32_t n_seq = promptsArr.Length();
    if (n_seq == 0) {
        Napi::TypeError::New(env, "inferBatch: prompts array must not be empty")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::vector<std::string> prompts(n_seq);
    std::vector<const char *> prompt_ptrs(n_seq);
    for (uint32_t i = 0; i < n_seq; i++) {
        prompts[i] = promptsArr.Get(i).As<Napi::String>().Utf8Value();
        prompt_ptrs[i] = prompts[i].c_str();
    }

    // Use first options entry for batch params (libhilum uses uniform params)
    GenContext gc;
    gc.params = hilum_gen_default_params();
    Napi::Array optsArr = info[3].As<Napi::Array>();
    if (optsArr.Length() > 0 && optsArr.Get(uint32_t(0)).IsObject())
        gc = parse_gen_context(optsArr.Get(uint32_t(0)).As<Napi::Object>());
    else
        gc.finalize();

    BatchTSFN tsfn = BatchTSFN::New(env, info[4].As<Napi::Function>(), "InferBatch", 0, 1);

    std::thread([model, ctx, prompt_ptrs, prompts, n_seq, gc, tsfn]() mutable {
        hilum_error err = hilum_generate_batch(model, ctx, prompt_ptrs.data(),
            static_cast<int32_t>(n_seq), gc.params,
            [](hilum_batch_event event, void * ud) -> bool {
                auto * tsfn_ptr = static_cast<BatchTSFN *>(ud);
                if (event.done) {
                    auto * data = new BatchStreamData{
                        event.seq_index, "", true, "",
                        event.finish_reason ? event.finish_reason : ""
                    };
                    tsfn_ptr->BlockingCall(data);
                } else {
                    auto * data = new BatchStreamData{
                        event.seq_index,
                        std::string(event.token, event.token_len),
                        false, "", ""
                    };
                    tsfn_ptr->BlockingCall(data);
                }
                return true;
            }, &tsfn);

        if (err != HILUM_OK && err != HILUM_ERR_CANCELLED) {
            auto * data = new BatchStreamData{0, "", false, hilum_error_str(err), ""};
            tsfn.BlockingCall(data);
        }

        auto * done = new BatchStreamData{-1, "", true, "", ""};
        tsfn.BlockingCall(done);
        tsfn.Release();
    }).detach();

    return env.Undefined();
}

// ── Vision ──────────────────────────────────────────────────────────────────

Napi::Value CreateMtmdContext(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsString()) {
        Napi::TypeError::New(env, "createMtmdContext(model, projectorPath, [options])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_mtmd_params params;
    params.use_gpu = true;
    params.n_threads = 0;

    if (info.Length() >= 3 && info[2].IsObject()) {
        Napi::Object opts = info[2].As<Napi::Object>();
        if (opts.Has("use_gpu"))   params.use_gpu = opts.Get("use_gpu").As<Napi::Boolean>().Value();
        if (opts.Has("n_threads")) params.n_threads = opts.Get("n_threads").As<Napi::Number>().Uint32Value();
    }

    hilum_mtmd * mtmd = nullptr;
    hilum_error err = hilum_mtmd_load(get_model(info[0]), js_string(info[1]).c_str(), params, &mtmd);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("createMtmdContext: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return Napi::External<hilum_mtmd>::New(env, mtmd);
}

Napi::Value FreeMtmdContext(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() >= 1 && info[0].IsExternal())
        hilum_mtmd_free(info[0].As<Napi::External<hilum_mtmd>>().Data());
    return env.Undefined();
}

Napi::Value SupportVision(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "supportVision(mtmdCtx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }
    return Napi::Boolean::New(env, hilum_mtmd_supports_vision(
        info[0].As<Napi::External<hilum_mtmd>>().Data()));
}

// Helper: build hilum_image array from JS Buffer array
static std::vector<hilum_image> build_images(const Napi::Array & arr) {
    std::vector<hilum_image> images;
    for (uint32_t i = 0; i < arr.Length(); i++) {
        Napi::Buffer<uint8_t> buf = arr.Get(i).As<Napi::Buffer<uint8_t>>();
        images.push_back({buf.Data(), buf.Length()});
    }
    return images;
}

Napi::Value InferSyncVision(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 5 || !info[0].IsExternal() || !info[1].IsExternal() ||
        !info[2].IsExternal() || !info[3].IsString() || !info[4].IsArray()) {
        Napi::TypeError::New(env, "inferSyncVision(model, ctx, mtmdCtx, prompt, imageBuffers, [options])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);
    hilum_context * ctx = get_ctx(info[1]);
    hilum_mtmd * mtmd = info[2].As<Napi::External<hilum_mtmd>>().Data();
    std::string prompt = js_string(info[3]);
    auto images = build_images(info[4].As<Napi::Array>());

    GenContext gc;
    gc.params = hilum_gen_default_params();
    if (info.Length() >= 6 && info[5].IsObject())
        gc = parse_gen_context(info[5].As<Napi::Object>());
    else
        gc.finalize();

    std::vector<char> buf(gc.params.max_tokens * 64 + 1024);
    int32_t generated = 0;

    hilum_error err = hilum_generate_vision(model, ctx, mtmd, prompt.c_str(),
        images.data(), static_cast<int32_t>(images.size()), gc.params,
        buf.data(), static_cast<int32_t>(buf.size()), &generated);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("inferSyncVision: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    return Napi::String::New(env, buf.data());
}

Napi::Value InferStreamVision(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 7 || !info[0].IsExternal() || !info[1].IsExternal() ||
        !info[2].IsExternal() || !info[3].IsString() || !info[4].IsArray() ||
        !info[6].IsFunction()) {
        Napi::TypeError::New(env, "inferStreamVision(model, ctx, mtmdCtx, prompt, imageBuffers, options, callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);
    hilum_context * ctx = get_ctx(info[1]);
    hilum_mtmd * mtmd = info[2].As<Napi::External<hilum_mtmd>>().Data();
    std::string prompt = js_string(info[3]);

    // Build images on main thread (needs JS Buffer access)
    auto images_vec = build_images(info[4].As<Napi::Array>());

    // Copy image data since the JS buffers may be GC'd before the thread runs
    struct ImageData { std::vector<uint8_t> data; };
    auto img_copies = std::make_shared<std::vector<ImageData>>();
    for (auto & img : images_vec) {
        img_copies->emplace_back();
        img_copies->back().data.assign(img.data, img.data + img.size);
    }

    GenContext gc;
    gc.params = hilum_gen_default_params();
    if (info[5].IsObject())
        gc = parse_gen_context(info[5].As<Napi::Object>());
    else
        gc.finalize();

    TSFN tsfn = TSFN::New(env, info[6].As<Napi::Function>(), "InferStreamVision", 0, 1);

    std::thread([model, ctx, mtmd, prompt, img_copies, gc, tsfn]() mutable {
        std::vector<hilum_image> images;
        for (auto & ic : *img_copies) {
            images.push_back({ic.data.data(), ic.data.size()});
        }

        hilum_error err = hilum_generate_vision_stream(model, ctx, mtmd, prompt.c_str(),
            images.data(), static_cast<int32_t>(images.size()), gc.params,
            [](const char * token, int32_t token_len, void * ud) -> bool {
                auto * tsfn_ptr = static_cast<TSFN *>(ud);
                auto * data = new StreamData{std::string(token, token_len), false, ""};
                tsfn_ptr->BlockingCall(data);
                return true;
            }, &tsfn);

        if (err != HILUM_OK && err != HILUM_ERR_CANCELLED) {
            auto * data = new StreamData{"", false, hilum_error_str(err)};
            tsfn.BlockingCall(data);
        }

        auto * done = new StreamData{"", true, ""};
        tsfn.BlockingCall(done);
        tsfn.Release();
    }).detach();

    return env.Undefined();
}

// ── Warmup ──────────────────────────────────────────────────────────────────

Napi::Value Warmup(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsExternal()) {
        Napi::TypeError::New(env, "warmup(model, ctx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_error err = hilum_warmup(get_model(info[0]), get_ctx(info[1]));
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("warmup: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
    }
    return env.Undefined();
}

// ── Performance metrics ─────────────────────────────────────────────────────

Napi::Value GetPerf(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsExternal()) {
        Napi::TypeError::New(env, "getPerf(ctx)").ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_perf_data perf = hilum_get_perf(get_ctx(info[0]));

    Napi::Object result = Napi::Object::New(env);
    result.Set("promptEvalMs",          Napi::Number::New(env, perf.prompt_eval_ms));
    result.Set("generationMs",          Napi::Number::New(env, perf.generation_ms));
    result.Set("promptTokens",          Napi::Number::New(env, perf.prompt_tokens));
    result.Set("generatedTokens",       Napi::Number::New(env, perf.generated_tokens));
    result.Set("promptTokensPerSec",    Napi::Number::New(env, perf.prompt_tokens_per_sec));
    result.Set("generatedTokensPerSec", Napi::Number::New(env, perf.generated_tokens_per_sec));
    return result;
}

// ── JSON Schema to Grammar ──────────────────────────────────────────────────

Napi::Value JsonSchemaToGrammar(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsString()) {
        Napi::TypeError::New(env, "jsonSchemaToGrammar(schemaJson: string)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string schema_json = js_string(info[0]);
    std::vector<char> buf(schema_json.size() * 8 + 4096);

    int32_t len = hilum_json_schema_to_grammar(schema_json.c_str(),
        buf.data(), static_cast<int32_t>(buf.size()));
    if (len <= 0) {
        if (len < 0) {
            buf.resize(-len);
            len = hilum_json_schema_to_grammar(schema_json.c_str(),
                buf.data(), static_cast<int32_t>(buf.size()));
        }
        if (len <= 0) {
            Napi::Error::New(env, "jsonSchemaToGrammar: conversion failed")
                .ThrowAsJavaScriptException();
            return env.Undefined();
        }
    }

    return Napi::String::New(env, buf.data(), len);
}

// ── Quantize ────────────────────────────────────────────────────────────────

struct QuantizeCallbackData {
    std::string error;
};

void QuantizeJsCallback(Napi::Env env, Napi::Function jsCallback, void *, QuantizeCallbackData * data) {
    if (data->error.empty()) {
        jsCallback.Call({env.Null()});
    } else {
        jsCallback.Call({Napi::Error::New(env, data->error).Value()});
    }
    delete data;
}

using QuantizeTSFN = Napi::TypedThreadSafeFunction<void, QuantizeCallbackData, QuantizeJsCallback>;

Napi::Value Quantize(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 4 || !info[0].IsString() || !info[1].IsString() ||
        !info[2].IsObject() || !info[3].IsFunction()) {
        Napi::TypeError::New(env, "quantize(inputPath, outputPath, options, callback)")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    std::string input_path  = js_string(info[0]);
    std::string output_path = js_string(info[1]);
    Napi::Object opts = info[2].As<Napi::Object>();

    hilum_quantize_params qp = hilum_quantize_default_params();
    if (opts.Has("ftype"))                  qp.ftype = opts.Get("ftype").As<Napi::Number>().Int32Value();
    if (opts.Has("nthread"))                qp.nthread = opts.Get("nthread").As<Napi::Number>().Int32Value();
    if (opts.Has("allow_requantize"))       qp.allow_requantize = opts.Get("allow_requantize").As<Napi::Boolean>().Value();
    if (opts.Has("quantize_output_tensor")) qp.quantize_output_tensor = opts.Get("quantize_output_tensor").As<Napi::Boolean>().Value();
    if (opts.Has("pure"))                   qp.pure = opts.Get("pure").As<Napi::Boolean>().Value();

    QuantizeTSFN tsfn = QuantizeTSFN::New(env, info[3].As<Napi::Function>(), "Quantize", 0, 1);

    std::thread([input_path, output_path, qp, tsfn]() {
        hilum_error err = hilum_quantize(input_path.c_str(), output_path.c_str(), qp);
        auto * data = new QuantizeCallbackData{};
        if (err != HILUM_OK) data->error = hilum_error_str(err);
        tsfn.BlockingCall(data);
        tsfn.Release();
    }).detach();

    return env.Undefined();
}

// ── Log Callback ────────────────────────────────────────────────────────────

struct LogCallbackData {
    std::string text;
    int level;
};

static void LogJsCallback(Napi::Env env, Napi::Function jsCallback, void *, LogCallbackData * data) {
    jsCallback.Call({
        Napi::Number::New(env, data->level),
        Napi::String::New(env, data->text)
    });
    delete data;
}

using LogTSFN = Napi::TypedThreadSafeFunction<void, LogCallbackData, LogJsCallback>;

static LogTSFN g_log_tsfn;
static bool g_log_tsfn_active = false;

Napi::Value SetLogCallback(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();

    if (g_log_tsfn_active) {
        hilum_log_set(nullptr, nullptr);
        g_log_tsfn.Release();
        g_log_tsfn_active = false;
    }

    if (info.Length() < 1 || info[0].IsNull() || info[0].IsUndefined())
        return env.Undefined();

    if (!info[0].IsFunction()) {
        Napi::TypeError::New(env, "setLogCallback: argument must be a function or null")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    g_log_tsfn = LogTSFN::New(env, info[0].As<Napi::Function>(), "LogCallback", 0, 1);
    g_log_tsfn_active = true;

    hilum_log_set([](hilum_log_level level, const char * text, void *) {
        if (!g_log_tsfn_active) return;
        auto * data = new LogCallbackData{std::string(text), static_cast<int>(level)};
        g_log_tsfn.BlockingCall(data);
    }, nullptr);

    return env.Undefined();
}

Napi::Value SetLogLevel(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 1 || !info[0].IsNumber()) {
        Napi::TypeError::New(env, "setLogLevel: argument must be a number")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }
    hilum_log_set_level(static_cast<hilum_log_level>(info[0].As<Napi::Number>().Int32Value()));
    return env.Undefined();
}

// ── Power state ─────────────────────────────────────────────────────────────

Napi::Value IsLowPowerMode(const Napi::CallbackInfo & info) {
#ifdef __APPLE__
    int val = 0;
    size_t len = sizeof(val);
    if (sysctlbyname("kern.lowpowermode", &val, &len, nullptr, 0) == 0) {
        return Napi::Boolean::New(info.Env(), val != 0);
    }
#endif
#ifdef __linux__
    std::ifstream status("/sys/class/power_supply/BAT0/status");
    if (status.good()) {
        std::string state;
        std::getline(status, state);
        return Napi::Boolean::New(info.Env(), state == "Discharging");
    }
#endif
    return Napi::Boolean::New(info.Env(), false);
}

// ── Optimal thread count ────────────────────────────────────────────────────

Napi::Value OptimalThreadCount(const Napi::CallbackInfo & info) {
    return Napi::Number::New(info.Env(), hilum_optimal_thread_count());
}

// ── Benchmark ───────────────────────────────────────────────────────────────

Napi::Value Benchmark(const Napi::CallbackInfo & info) {
    Napi::Env env = info.Env();
    if (info.Length() < 2 || !info[0].IsExternal() || !info[1].IsExternal()) {
        Napi::TypeError::New(env, "benchmark(model, ctx, [options])")
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    hilum_model * model = get_model(info[0]);
    hilum_context * ctx = get_ctx(info[1]);

    hilum_benchmark_params bp = hilum_benchmark_default_params();
    if (info.Length() >= 3 && info[2].IsObject()) {
        Napi::Object opts = info[2].As<Napi::Object>();
        if (opts.Has("promptTokens"))   bp.prompt_tokens   = opts.Get("promptTokens").As<Napi::Number>().Int32Value();
        if (opts.Has("generateTokens")) bp.generate_tokens = opts.Get("generateTokens").As<Napi::Number>().Int32Value();
        if (opts.Has("iterations"))     bp.iterations      = opts.Get("iterations").As<Napi::Number>().Int32Value();
    }

    hilum_benchmark_result result = {};
    hilum_error err = hilum_benchmark(model, ctx, bp, &result);
    if (err != HILUM_OK) {
        Napi::Error::New(env, std::string("benchmark: ") + hilum_error_str(err))
            .ThrowAsJavaScriptException();
        return env.Undefined();
    }

    Napi::Object out = Napi::Object::New(env);
    out.Set("promptTokensPerSec",    Napi::Number::New(env, result.prompt_tokens_per_sec));
    out.Set("generatedTokensPerSec", Napi::Number::New(env, result.generated_tokens_per_sec));
    out.Set("ttftMs",                Napi::Number::New(env, result.ttft_ms));
    out.Set("totalMs",               Napi::Number::New(env, result.total_ms));
    out.Set("iterations",            Napi::Number::New(env, result.iterations));
    return out;
}

// ── Module Init ─────────────────────────────────────────────────────────────

Napi::Object Init(Napi::Env env, Napi::Object exports) {
    exports.Set("backendInfo",            Napi::Function::New(env, BackendInfo));
    exports.Set("backendVersion",         Napi::Function::New(env, BackendVersion));
    exports.Set("loadModel",             Napi::Function::New(env, LoadModel));
    exports.Set("loadModelFromBuffer",   Napi::Function::New(env, LoadModelFromBuffer));
    exports.Set("getModelSize",          Napi::Function::New(env, GetModelSize));
    exports.Set("createContext",         Napi::Function::New(env, CreateContext));
    exports.Set("freeModel",             Napi::Function::New(env, FreeModel));
    exports.Set("freeContext",           Napi::Function::New(env, FreeContext));
    exports.Set("getContextSize",        Napi::Function::New(env, GetContextSize));
    exports.Set("tokenize",              Napi::Function::New(env, Tokenize));
    exports.Set("detokenize",            Napi::Function::New(env, Detokenize));
    exports.Set("applyChatTemplate",     Napi::Function::New(env, ApplyChatTemplate));
    exports.Set("inferSync",             Napi::Function::New(env, InferSync));
    exports.Set("inferStream",           Napi::Function::New(env, InferStream));
    exports.Set("kvCacheClear",          Napi::Function::New(env, KvCacheClear));
    exports.Set("setLogCallback",        Napi::Function::New(env, SetLogCallback));
    exports.Set("setLogLevel",           Napi::Function::New(env, SetLogLevel));
    exports.Set("createMtmdContext",     Napi::Function::New(env, CreateMtmdContext));
    exports.Set("freeMtmdContext",       Napi::Function::New(env, FreeMtmdContext));
    exports.Set("supportVision",         Napi::Function::New(env, SupportVision));
    exports.Set("inferSyncVision",       Napi::Function::New(env, InferSyncVision));
    exports.Set("inferStreamVision",     Napi::Function::New(env, InferStreamVision));
    exports.Set("jsonSchemaToGrammar",   Napi::Function::New(env, JsonSchemaToGrammar));
    exports.Set("getEmbeddingDimension", Napi::Function::New(env, GetEmbeddingDimension));
    exports.Set("createEmbeddingContext", Napi::Function::New(env, CreateEmbeddingContext));
    exports.Set("embed",                 Napi::Function::New(env, Embed));
    exports.Set("embedBatch",            Napi::Function::New(env, EmbedBatch));
    exports.Set("quantize",              Napi::Function::New(env, Quantize));
    exports.Set("inferBatch",            Napi::Function::New(env, InferBatch));
    exports.Set("warmup",               Napi::Function::New(env, Warmup));
    exports.Set("getPerf",              Napi::Function::New(env, GetPerf));
    exports.Set("isLowPowerMode",       Napi::Function::New(env, IsLowPowerMode));
    exports.Set("optimalThreadCount",   Napi::Function::New(env, OptimalThreadCount));
    exports.Set("benchmark",            Napi::Function::New(env, Benchmark));
    return exports;
}

NODE_API_MODULE(hilum_native, Init)
