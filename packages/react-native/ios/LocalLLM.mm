#import "LocalLLM.h"

#import <React/RCTBridge.h>
#import <React/RCTLog.h>

#include <string>
#include <unordered_map>
#include <mutex>
#include <set>
#include <vector>
#include <cstdint>
#include <cmath>
#include <atomic>

#include "llama.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#import <Metal/Metal.h>
#import <os/proc.h>

// ── UUID generation ──────────────────────────────────────────────────────────

static NSString *generateUUID() {
  return [[NSUUID UUID] UUIDString];
}

// ── Handle maps ──────────────────────────────────────────────────────────────

static std::mutex g_mutex;

static std::unordered_map<std::string, llama_model *> g_models;
static std::unordered_map<std::string, llama_context *> g_contexts;
static std::unordered_map<std::string, mtmd_context *> g_mtmd_contexts;

// Track live handles to prevent double-free
static std::set<void *> g_live_handles;

static void register_handle(void *ptr) {
  std::lock_guard<std::mutex> lock(g_mutex);
  g_live_handles.insert(ptr);
}

static bool unregister_handle(void *ptr) {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_live_handles.erase(ptr) > 0;
}

// ── Stream cancellation ──────────────────────────────────────────────────────

static std::mutex g_cancel_mutex;
static std::set<std::string> g_cancel_set;

static void request_cancel(const std::string &ctxId) {
  std::lock_guard<std::mutex> lock(g_cancel_mutex);
  g_cancel_set.insert(ctxId);
}

static bool is_cancelled(const std::string &ctxId) {
  std::lock_guard<std::mutex> lock(g_cancel_mutex);
  return g_cancel_set.count(ctxId) > 0;
}

static void clear_cancel(const std::string &ctxId) {
  std::lock_guard<std::mutex> lock(g_cancel_mutex);
  g_cancel_set.erase(ctxId);
}

// ── Log state (static so the C callback can access without capturing `self`) ─

static std::atomic<int> g_log_min_level{2};          // default: info (GGML_LOG_LEVEL_INFO)
static std::atomic<bool> g_log_events_enabled{false};
static __weak LocalLLM *g_log_module = nil;

// ── Sampler creation ─────────────────────────────────────────────────────────

struct SamplerParams {
  int32_t max_tokens = 256;
  float temperature = 0.7f;
  float top_p = 0.95f;
  int32_t top_k = 40;
  float repeat_penalty = 1.1f;
  float frequency_penalty = 0.0f;
  float presence_penalty = 0.0f;
  int32_t seed = -1;
  std::string grammar;
  std::string grammar_root;
  int32_t n_past = 0;
};

static SamplerParams parse_sampler_params(NSDictionary *options) {
  SamplerParams p;
  if (options[@"max_tokens"])          p.max_tokens = [options[@"max_tokens"] intValue];
  if (options[@"temperature"])         p.temperature = [options[@"temperature"] floatValue];
  if (options[@"top_p"])               p.top_p = [options[@"top_p"] floatValue];
  if (options[@"top_k"])               p.top_k = [options[@"top_k"] intValue];
  if (options[@"repeat_penalty"])      p.repeat_penalty = [options[@"repeat_penalty"] floatValue];
  if (options[@"frequency_penalty"])   p.frequency_penalty = [options[@"frequency_penalty"] floatValue];
  if (options[@"presence_penalty"])    p.presence_penalty = [options[@"presence_penalty"] floatValue];
  if (options[@"seed"])                p.seed = [options[@"seed"] intValue];
  if (options[@"grammar"])             p.grammar = [options[@"grammar"] UTF8String];
  if (options[@"grammar_root"])        p.grammar_root = [options[@"grammar_root"] UTF8String];
  if (options[@"n_past"])              p.n_past = [options[@"n_past"] intValue];
  return p;
}

static llama_sampler *create_sampler(const SamplerParams &p, const llama_model *model) {
  auto *smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());

  if (p.repeat_penalty != 1.0f || p.frequency_penalty != 0.0f || p.presence_penalty != 0.0f) {
    llama_sampler_chain_add(smpl,
      llama_sampler_init_penalties(
        llama_model_n_ctx_train(model),
        p.repeat_penalty,
        p.frequency_penalty,
        p.presence_penalty
      ));
  }

  if (p.temperature <= 0.0f) {
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
  } else {
    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(p.top_k));
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(p.top_p, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(p.temperature));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(p.seed));
  }

  if (!p.grammar.empty()) {
    llama_sampler_chain_add(smpl,
      llama_sampler_init_grammar(
        llama_model_get_vocab(model),
        p.grammar.c_str(),
        p.grammar_root.empty() ? "root" : p.grammar_root.c_str()
      ));
  }

  return smpl;
}

// ── Token to string helper ───────────────────────────────────────────────────

static std::string token_to_piece(const llama_model *model, llama_token token) {
  char buf[256];
  int n = llama_token_to_piece(llama_model_get_vocab(model), token, buf, sizeof(buf), 0, true);
  if (n < 0) {
    std::string result(-(int)n, '\0');
    llama_token_to_piece(llama_model_get_vocab(model), token, result.data(), result.size(), 0, true);
    return result;
  }
  return std::string(buf, n);
}

// ── Base64 decoding ──────────────────────────────────────────────────────────

static std::vector<uint8_t> decode_base64(NSString *base64) {
  NSData *data = [[NSData alloc] initWithBase64EncodedString:base64 options:0];
  if (!data) return {};
  const uint8_t *bytes = (const uint8_t *)[data bytes];
  return std::vector<uint8_t>(bytes, bytes + [data length]);
}

// ── Inference dispatch queue ─────────────────────────────────────────────────

static dispatch_queue_t inference_queue() {
  static dispatch_queue_t q = dispatch_queue_create("com.hilum.llm.inference", DISPATCH_QUEUE_SERIAL);
  return q;
}

// ── Download session management ──────────────────────────────────────────────

@interface LLMDownloadDelegate : NSObject <NSURLSessionDownloadDelegate>
@property (nonatomic, weak) LocalLLM *module;
@property (nonatomic, strong) NSMutableDictionary<NSString *, NSString *> *destPaths;
@end

@implementation LLMDownloadDelegate

- (instancetype)initWithModule:(LocalLLM *)module {
  self = [super init];
  if (self) {
    _module = module;
    _destPaths = [NSMutableDictionary new];
  }
  return self;
}

- (void)URLSession:(NSURLSession *)session
      downloadTask:(NSURLSessionDownloadTask *)downloadTask
      didWriteData:(int64_t)bytesWritten
 totalBytesWritten:(int64_t)totalBytesWritten
totalBytesExpectedToWrite:(int64_t)totalBytesExpectedToWrite {
  NSString *url = downloadTask.originalRequest.URL.absoluteString;
  double percent = totalBytesExpectedToWrite > 0
    ? (double)totalBytesWritten / (double)totalBytesExpectedToWrite * 100.0
    : 0.0;
  [_module sendEventWithName:@"onDownloadProgress" body:@{
    @"url": url ?: @"",
    @"downloaded": @(totalBytesWritten),
    @"total": @(totalBytesExpectedToWrite),
    @"percent": @(percent),
  }];
}

- (void)URLSession:(NSURLSession *)session
      downloadTask:(NSURLSessionDownloadTask *)downloadTask
didFinishDownloadingToURL:(NSURL *)location {
  NSString *url = downloadTask.originalRequest.URL.absoluteString;
  NSString *destPath = _destPaths[url];
  if (destPath) {
    NSError *error = nil;
    NSFileManager *fm = [NSFileManager defaultManager];
    // Remove existing file if present
    [fm removeItemAtPath:destPath error:nil];
    // Create parent directory
    [fm createDirectoryAtPath:[destPath stringByDeletingLastPathComponent]
      withIntermediateDirectories:YES attributes:nil error:nil];
    [fm moveItemAtURL:location toURL:[NSURL fileURLWithPath:destPath] error:&error];
    if (error) {
      [_module sendEventWithName:@"onDownloadError" body:@{
        @"url": url ?: @"",
        @"error": error.localizedDescription ?: @"Move failed",
        @"resumable": @NO,
      }];
      return;
    }
  }
  [_module sendEventWithName:@"onDownloadComplete" body:@{
    @"url": url ?: @"",
  }];
}

- (void)URLSession:(NSURLSession *)session
              task:(NSURLSessionTask *)task
didCompleteWithError:(NSError *)error {
  if (!error) return;
  NSString *url = task.originalRequest.URL.absoluteString;
  BOOL resumable = error.userInfo[NSURLSessionDownloadTaskResumeData] != nil;
  [_module sendEventWithName:@"onDownloadError" body:@{
    @"url": url ?: @"",
    @"error": error.localizedDescription ?: @"Download failed",
    @"resumable": @(resumable),
  }];
}

@end

// ── Module implementation ────────────────────────────────────────────────────

@implementation LocalLLM {
  NSURLSession *_downloadSession;
  LLMDownloadDelegate *_downloadDelegate;
  bool _hasListeners;
}

RCT_EXPORT_MODULE()

+ (BOOL)requiresMainQueueSetup {
  return NO;
}

- (instancetype)init {
  self = [super init];
  if (self) {
    _hasListeners = NO;
    _downloadDelegate = [[LLMDownloadDelegate alloc] initWithModule:self];
    NSURLSessionConfiguration *config =
      [NSURLSessionConfiguration backgroundSessionConfigurationWithIdentifier:@"com.hilum.llm.downloads"];
    config.sessionSendsLaunchEvents = YES;
    _downloadSession = [NSURLSession sessionWithConfiguration:config
                                                    delegate:_downloadDelegate
                                               delegateQueue:nil];
  }
  return self;
}

- (NSArray<NSString *> *)supportedEvents {
  return @[
    @"onToken",
    @"onBatchToken",
    @"onQuantizeComplete",
    @"onLog",
    @"onDownloadProgress",
    @"onDownloadComplete",
    @"onDownloadError",
  ];
}

- (void)startObserving { _hasListeners = YES; }
- (void)stopObserving  { _hasListeners = NO; }

// ── Backend info ─────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(backendInfo) {
  return @(llama_print_system_info());
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(backendVersion) {
  // Use LLAMA_BUILD_NUMBER if available, otherwise a static version
  return @"1.0.0";
}

// ── Model lifecycle ──────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(loadModel:(NSString *)path
                  options:(NSDictionary *)options
                  resolve:(RCTPromiseResolveBlock)resolve
                  reject:(RCTPromiseRejectBlock)reject) {
  dispatch_async(inference_queue(), ^{
    // RAM guard
    uint64_t available = os_proc_available_memory();
    uint64_t minimumRAM = 512 * 1024 * 1024;  // 512 MB absolute floor

    if (available < minimumRAM) {
      reject(@"E_INSUFFICIENT_MEMORY",
        [NSString stringWithFormat:
          @"Insufficient memory to load model. Available: %llu MB, minimum: %llu MB. "
          @"Close other apps or use a smaller quantized model.",
          available / (1024 * 1024), minimumRAM / (1024 * 1024)],
        nil);
      return;
    }

    int n_gpu_layers = options[@"n_gpu_layers"] ? [options[@"n_gpu_layers"] intValue] : 999;
    bool use_mmap = options[@"use_mmap"] ? [options[@"use_mmap"] boolValue] : true;

    llama_model_params params = llama_model_default_params();
    params.n_gpu_layers = n_gpu_layers;
    params.use_mmap = use_mmap;

    llama_model *model = llama_model_load_from_file([path UTF8String], params);
    if (!model) {
      reject(@"E_MODEL_LOAD", @"Failed to load model", nil);
      return;
    }

    register_handle(model);
    NSString *modelId = generateUUID();
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_models[[modelId UTF8String]] = model;
    }
    resolve(modelId);
  });
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getModelSize:(NSString *)modelId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @(0);
  return @((double)llama_model_size(it->second));
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(freeModel:(NSString *)modelId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it != g_models.end()) {
    if (unregister_handle(it->second)) {
      llama_model_free(it->second);
    }
    g_models.erase(it);
  }
  return nil;
}

// ── Context lifecycle ────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(createContext:(NSString *)modelId
                                        options:(NSDictionary *)options) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  llama_context_params params = llama_context_default_params();
  if (options[@"n_ctx"])        params.n_ctx = [options[@"n_ctx"] intValue];
  if (options[@"n_batch"])      params.n_batch = [options[@"n_batch"] intValue];
  if (options[@"n_threads"])    params.n_threads = [options[@"n_threads"] intValue];
  if (options[@"n_seq_max"])    params.n_seq_max = [options[@"n_seq_max"] intValue];
  if (options[@"flash_attn_type"]) params.flash_attn = [options[@"flash_attn_type"] intValue] > 0;
  if (options[@"type_k"])       params.type_k = (enum ggml_type)[options[@"type_k"] intValue];
  if (options[@"type_v"])       params.type_v = (enum ggml_type)[options[@"type_v"] intValue];

  llama_context *ctx = llama_init_from_model(it->second, params);
  if (!ctx) return @"";

  register_handle(ctx);
  NSString *ctxId = generateUUID();
  g_contexts[[ctxId UTF8String]] = ctx;
  return ctxId;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getContextSize:(NSString *)contextId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_contexts.find([contextId UTF8String]);
  if (it == g_contexts.end()) return @(0);
  return @((int)llama_n_ctx(it->second));
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(freeContext:(NSString *)contextId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_contexts.find([contextId UTF8String]);
  if (it != g_contexts.end()) {
    if (unregister_handle(it->second)) {
      llama_free(it->second);
    }
    g_contexts.erase(it);
  }
  return nil;
}

// ── KV cache ─────────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(kvCacheClear:(NSString *)contextId
                                        fromPos:(double)fromPos) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_contexts.find([contextId UTF8String]);
  if (it != g_contexts.end()) {
    llama_kv_cache_seq_rm(it->second, 0, (int)fromPos, -1);
  }
  return nil;
}

// ── Tokenization ─────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(tokenize:(NSString *)modelId
                                          text:(NSString *)text
                                    addSpecial:(BOOL)addSpecial
                                  parseSpecial:(BOOL)parseSpecial) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @[];

  const char *ctext = [text UTF8String];
  int text_len = (int)strlen(ctext);
  const llama_vocab *vocab = llama_model_get_vocab(it->second);

  std::vector<llama_token> tokens(text_len + 16);
  int n = llama_tokenize(vocab, ctext, text_len, tokens.data(), (int)tokens.size(), addSpecial, parseSpecial);
  if (n < 0) {
    tokens.resize(-n);
    n = llama_tokenize(vocab, ctext, text_len, tokens.data(), (int)tokens.size(), addSpecial, parseSpecial);
  }
  tokens.resize(n);

  NSMutableArray *result = [NSMutableArray arrayWithCapacity:n];
  for (int i = 0; i < n; i++) {
    [result addObject:@(tokens[i])];
  }
  return result;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(detokenize:(NSString *)modelId
                                          tokens:(NSArray<NSNumber *> *)tokens) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  std::string result;
  for (NSNumber *tok in tokens) {
    result += token_to_piece(it->second, [tok intValue]);
  }
  return [NSString stringWithUTF8String:result.c_str()];
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(applyChatTemplate:(NSString *)modelId
                                            messages:(NSArray<NSDictionary *> *)messages
                                        addAssistant:(BOOL)addAssistant) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  std::vector<llama_chat_msg> chat_msgs;
  for (NSDictionary *msg in messages) {
    llama_chat_msg m;
    m.role = [msg[@"role"] UTF8String];
    m.content = [msg[@"content"] UTF8String];
    chat_msgs.push_back(m);
  }

  std::string result(4096, '\0');
  int n = llama_chat_apply_template(
    llama_model_chat_template(it->second, nullptr),
    chat_msgs.data(), chat_msgs.size(),
    addAssistant,
    result.data(), (int)result.size()
  );
  if (n > (int)result.size()) {
    result.resize(n);
    llama_chat_apply_template(
      llama_model_chat_template(it->second, nullptr),
      chat_msgs.data(), chat_msgs.size(),
      addAssistant,
      result.data(), (int)result.size()
    );
  }
  result.resize(n);
  return [NSString stringWithUTF8String:result.c_str()];
}

// ── Text inference ───────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(generate:(NSString *)modelId
              contextId:(NSString *)contextId
                 prompt:(NSString *)prompt
                options:(NSDictionary *)options
                resolve:(RCTPromiseResolveBlock)resolve
                 reject:(RCTPromiseRejectBlock)reject) {
  dispatch_async(inference_queue(), ^{
    llama_model *model;
    llama_context *ctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find([contextId UTF8String]);
      if (mi == g_models.end() || ci == g_contexts.end()) {
        reject(@"E_NOT_FOUND", @"Model or context not found", nil);
        return;
      }
      model = mi->second;
      ctx = ci->second;
    }

    SamplerParams sp = parse_sampler_params(options);
    const char *cprompt = [prompt UTF8String];
    const llama_vocab *vocab = llama_model_get_vocab(model);

    // Tokenize prompt
    int prompt_len = (int)strlen(cprompt);
    std::vector<llama_token> tokens(prompt_len + 16);
    int n_tokens = llama_tokenize(vocab, cprompt, prompt_len, tokens.data(), (int)tokens.size(), true, true);
    if (n_tokens < 0) {
      tokens.resize(-n_tokens);
      n_tokens = llama_tokenize(vocab, cprompt, prompt_len, tokens.data(), (int)tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // Eval prompt
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = sp.n_past; i < n_tokens; i++) {
      llama_batch_add(batch, tokens[i], i, {0}, i == n_tokens - 1);
    }
    if (llama_decode(ctx, batch) != 0) {
      llama_batch_free(batch);
      reject(@"E_DECODE", @"Failed to decode prompt", nil);
      return;
    }
    llama_batch_free(batch);

    // Sample loop
    llama_sampler *smpl = create_sampler(sp, model);
    std::string result;

    for (int i = 0; i < sp.max_tokens; i++) {
      llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

      if (llama_vocab_is_eog(vocab, new_token)) break;

      result += token_to_piece(model, new_token);

      // Eval the new token
      llama_batch single = llama_batch_init(1, 0, 1);
      llama_batch_add(single, new_token, n_tokens + i, {0}, true);
      if (llama_decode(ctx, single) != 0) {
        llama_batch_free(single);
        break;
      }
      llama_batch_free(single);
    }

    llama_sampler_free(smpl);
    resolve([NSString stringWithUTF8String:result.c_str()]);
  });
}

RCT_EXPORT_METHOD(startStream:(NSString *)modelId
               contextId:(NSString *)contextId
                  prompt:(NSString *)prompt
                 options:(NSDictionary *)options) {
  std::string ctxIdStr = [contextId UTF8String];
  clear_cancel(ctxIdStr);

  dispatch_async(inference_queue(), ^{
    llama_model *model;
    llama_context *ctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find(ctxIdStr);
      if (mi == g_models.end() || ci == g_contexts.end()) {
        [self sendEventWithName:@"onToken" body:@{
          @"contextId": contextId, @"done": @YES, @"error": @"Model or context not found"
        }];
        return;
      }
      model = mi->second;
      ctx = ci->second;
    }

    SamplerParams sp = parse_sampler_params(options);
    const char *cprompt = [prompt UTF8String];
    const llama_vocab *vocab = llama_model_get_vocab(model);

    // Tokenize prompt
    int prompt_len = (int)strlen(cprompt);
    std::vector<llama_token> tokens(prompt_len + 16);
    int n_tokens = llama_tokenize(vocab, cprompt, prompt_len, tokens.data(), (int)tokens.size(), true, true);
    if (n_tokens < 0) {
      tokens.resize(-n_tokens);
      n_tokens = llama_tokenize(vocab, cprompt, prompt_len, tokens.data(), (int)tokens.size(), true, true);
    }
    tokens.resize(n_tokens);

    // Eval prompt
    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = sp.n_past; i < n_tokens; i++) {
      llama_batch_add(batch, tokens[i], i, {0}, i == n_tokens - 1);
    }
    if (llama_decode(ctx, batch) != 0) {
      llama_batch_free(batch);
      [self sendEventWithName:@"onToken" body:@{
        @"contextId": contextId, @"done": @YES, @"error": @"Failed to decode prompt"
      }];
      return;
    }
    llama_batch_free(batch);

    // Sample loop
    llama_sampler *smpl = create_sampler(sp, model);

    for (int i = 0; i < sp.max_tokens; i++) {
      if (is_cancelled(ctxIdStr)) break;

      llama_token new_token = llama_sampler_sample(smpl, ctx, -1);

      if (llama_vocab_is_eog(vocab, new_token)) break;

      std::string piece = token_to_piece(model, new_token);
      [self sendEventWithName:@"onToken" body:@{
        @"contextId": contextId,
        @"token": [NSString stringWithUTF8String:piece.c_str()],
        @"done": @NO,
      }];

      llama_batch single = llama_batch_init(1, 0, 1);
      llama_batch_add(single, new_token, n_tokens + i, {0}, true);
      if (llama_decode(ctx, single) != 0) {
        llama_batch_free(single);
        break;
      }
      llama_batch_free(single);
    }

    llama_sampler_free(smpl);
    clear_cancel(ctxIdStr);
    [self sendEventWithName:@"onToken" body:@{
      @"contextId": contextId, @"done": @YES
    }];
  });
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(stopStream:(NSString *)contextId) {
  request_cancel([contextId UTF8String]);
  return nil;
}

// ── Vision ───────────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(loadProjector:(NSString *)modelId
                                              path:(NSString *)path
                                           options:(NSDictionary *)options) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  mtmd_context_params mparams = mtmd_context_default_params();
  mparams.use_gpu = options[@"use_gpu"] ? [options[@"use_gpu"] boolValue] : true;
  if (options[@"n_threads"]) mparams.n_threads = [options[@"n_threads"] intValue];

  mtmd_context *mctx = mtmd_init_from_file([path UTF8String], it->second, mparams);
  if (!mctx) return @"";

  register_handle(mctx);
  NSString *mtmdId = generateUUID();
  g_mtmd_contexts[[mtmdId UTF8String]] = mctx;
  return mtmdId;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(supportVision:(NSString *)mtmdId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_mtmd_contexts.find([mtmdId UTF8String]);
  if (it == g_mtmd_contexts.end()) return @NO;
  return @(mtmd_support_vision(it->second));
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(freeMtmdContext:(NSString *)mtmdId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_mtmd_contexts.find([mtmdId UTF8String]);
  if (it != g_mtmd_contexts.end()) {
    if (unregister_handle(it->second)) {
      mtmd_free(it->second);
    }
    g_mtmd_contexts.erase(it);
  }
  return nil;
}

RCT_EXPORT_METHOD(generateVision:(NSString *)modelId
                     contextId:(NSString *)contextId
                        mtmdId:(NSString *)mtmdId
                        prompt:(NSString *)prompt
                  imageBase64s:(NSArray<NSString *> *)imageBase64s
                       options:(NSDictionary *)options
                       resolve:(RCTPromiseResolveBlock)resolve
                        reject:(RCTPromiseRejectBlock)reject) {
  dispatch_async(inference_queue(), ^{
    llama_model *model;
    llama_context *ctx;
    mtmd_context *mctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find([contextId UTF8String]);
      auto vi = g_mtmd_contexts.find([mtmdId UTF8String]);
      if (mi == g_models.end() || ci == g_contexts.end() || vi == g_mtmd_contexts.end()) {
        reject(@"E_NOT_FOUND", @"Model, context, or vision context not found", nil);
        return;
      }
      model = mi->second;
      ctx = ci->second;
      mctx = vi->second;
    }

    // Decode base64 images
    std::vector<mtmd_bitmap> bitmaps;
    for (NSString *b64 in imageBase64s) {
      std::vector<uint8_t> imgData = decode_base64(b64);
      if (imgData.empty()) continue;
      mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_buf(imgData.data(), imgData.size());
      if (bmp) bitmaps.push_back(*bmp);
    }

    // Tokenize with vision
    SamplerParams sp = parse_sampler_params(options);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();

    if (mtmd_tokenize(mctx, chunks, [prompt UTF8String], bitmaps.data(), bitmaps.size()) != 0) {
      mtmd_input_chunks_free(chunks);
      reject(@"E_VISION_TOKENIZE", @"Failed to tokenize vision input", nil);
      return;
    }

    // Eval chunks
    if (mtmd_helper_eval(mctx, ctx, chunks, llama_n_ctx(ctx), 0) != 0) {
      mtmd_input_chunks_free(chunks);
      reject(@"E_VISION_EVAL", @"Failed to evaluate vision input", nil);
      return;
    }

    int n_past = mtmd_helper_get_n_pos(chunks);
    mtmd_input_chunks_free(chunks);

    // Sample
    llama_sampler *smpl = create_sampler(sp, model);
    std::string result;

    for (int i = 0; i < sp.max_tokens; i++) {
      llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
      if (llama_vocab_is_eog(vocab, new_token)) break;

      result += token_to_piece(model, new_token);

      llama_batch single = llama_batch_init(1, 0, 1);
      llama_batch_add(single, new_token, n_past + i, {0}, true);
      if (llama_decode(ctx, single) != 0) {
        llama_batch_free(single);
        break;
      }
      llama_batch_free(single);
    }

    llama_sampler_free(smpl);
    resolve([NSString stringWithUTF8String:result.c_str()]);
  });
}

RCT_EXPORT_METHOD(startStreamVision:(NSString *)modelId
                        contextId:(NSString *)contextId
                           mtmdId:(NSString *)mtmdId
                           prompt:(NSString *)prompt
                     imageBase64s:(NSArray<NSString *> *)imageBase64s
                          options:(NSDictionary *)options) {
  std::string ctxIdStr = [contextId UTF8String];
  clear_cancel(ctxIdStr);

  dispatch_async(inference_queue(), ^{
    llama_model *model;
    llama_context *ctx;
    mtmd_context *mctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find(ctxIdStr);
      auto vi = g_mtmd_contexts.find([mtmdId UTF8String]);
      if (mi == g_models.end() || ci == g_contexts.end() || vi == g_mtmd_contexts.end()) {
        [self sendEventWithName:@"onToken" body:@{
          @"contextId": contextId, @"done": @YES, @"error": @"Not found"
        }];
        return;
      }
      model = mi->second;
      ctx = ci->second;
      mctx = vi->second;
    }

    // Decode base64 images
    std::vector<mtmd_bitmap> bitmaps;
    for (NSString *b64 in imageBase64s) {
      std::vector<uint8_t> imgData = decode_base64(b64);
      if (imgData.empty()) continue;
      mtmd_bitmap *bmp = mtmd_helper_bitmap_init_from_buf(imgData.data(), imgData.size());
      if (bmp) bitmaps.push_back(*bmp);
    }

    SamplerParams sp = parse_sampler_params(options);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();

    if (mtmd_tokenize(mctx, chunks, [prompt UTF8String], bitmaps.data(), bitmaps.size()) != 0) {
      mtmd_input_chunks_free(chunks);
      [self sendEventWithName:@"onToken" body:@{
        @"contextId": contextId, @"done": @YES, @"error": @"Vision tokenize failed"
      }];
      return;
    }

    if (mtmd_helper_eval(mctx, ctx, chunks, llama_n_ctx(ctx), 0) != 0) {
      mtmd_input_chunks_free(chunks);
      [self sendEventWithName:@"onToken" body:@{
        @"contextId": contextId, @"done": @YES, @"error": @"Vision eval failed"
      }];
      return;
    }

    int n_past = mtmd_helper_get_n_pos(chunks);
    mtmd_input_chunks_free(chunks);

    llama_sampler *smpl = create_sampler(sp, model);

    for (int i = 0; i < sp.max_tokens; i++) {
      if (is_cancelled(ctxIdStr)) break;

      llama_token new_token = llama_sampler_sample(smpl, ctx, -1);
      if (llama_vocab_is_eog(vocab, new_token)) break;

      std::string piece = token_to_piece(model, new_token);
      [self sendEventWithName:@"onToken" body:@{
        @"contextId": contextId,
        @"token": [NSString stringWithUTF8String:piece.c_str()],
        @"done": @NO,
      }];

      llama_batch single = llama_batch_init(1, 0, 1);
      llama_batch_add(single, new_token, n_past + i, {0}, true);
      if (llama_decode(ctx, single) != 0) {
        llama_batch_free(single);
        break;
      }
      llama_batch_free(single);
    }

    llama_sampler_free(smpl);
    clear_cancel(ctxIdStr);
    [self sendEventWithName:@"onToken" body:@{
      @"contextId": contextId, @"done": @YES
    }];
  });
}

// ── Grammar ──────────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(jsonSchemaToGrammar:(NSString *)schemaJson) {
  try {
    auto schema = nlohmann::ordered_json::parse([schemaJson UTF8String]);
    std::string grammar = json_schema_to_grammar(schema);
    return [NSString stringWithUTF8String:grammar.c_str()];
  } catch (...) {
    return @"";
  }
}

// ── Embeddings ───────────────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getEmbeddingDimension:(NSString *)modelId) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @(0);
  return @((int)llama_model_n_embd(it->second));
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(createEmbeddingContext:(NSString *)modelId
                                                      options:(NSDictionary *)options) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_models.find([modelId UTF8String]);
  if (it == g_models.end()) return @"";

  llama_context_params params = llama_context_default_params();
  params.embeddings = true;
  if (options[@"n_ctx"])        params.n_ctx = [options[@"n_ctx"] intValue];
  if (options[@"n_batch"])      params.n_batch = [options[@"n_batch"] intValue];
  if (options[@"n_threads"])    params.n_threads = [options[@"n_threads"] intValue];
  if (options[@"pooling_type"]) params.pooling_type = (enum llama_pooling_type)[options[@"pooling_type"] intValue];

  llama_context *ctx = llama_init_from_model(it->second, params);
  if (!ctx) return @"";

  register_handle(ctx);
  NSString *ctxId = generateUUID();
  g_contexts[[ctxId UTF8String]] = ctx;
  return ctxId;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(embed:(NSString *)contextId
                                      modelId:(NSString *)modelId
                                       tokens:(NSArray<NSNumber *> *)tokens) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ci = g_contexts.find([contextId UTF8String]);
  auto mi = g_models.find([modelId UTF8String]);
  if (ci == g_contexts.end() || mi == g_models.end()) {
    @throw [NSException exceptionWithName:@"E_INVALID_HANDLE"
                                   reason:@"Invalid context or model ID for embed()"
                                 userInfo:nil];
  }

  llama_context *ctx = ci->second;

  // Build batch
  int n = (int)tokens.count;
  llama_batch batch = llama_batch_init(n, 0, 1);
  for (int i = 0; i < n; i++) {
    llama_batch_add(batch, [tokens[i] intValue], i, {0}, true);
  }

  if (llama_encode(ctx, batch) != 0) {
    llama_batch_free(batch);
    return @[];
  }
  llama_batch_free(batch);

  // Extract embeddings
  int n_embd = llama_model_n_embd(mi->second);
  const float *embd = llama_get_embeddings_seq(ctx, 0);
  if (!embd) return @[];

  // L2 normalize
  float norm = 0.0f;
  for (int i = 0; i < n_embd; i++) norm += embd[i] * embd[i];
  norm = sqrtf(norm);

  NSMutableArray *result = [NSMutableArray arrayWithCapacity:n_embd];
  for (int i = 0; i < n_embd; i++) {
    [result addObject:@(norm > 0.0f ? embd[i] / norm : 0.0f)];
  }
  return result;
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(embedBatch:(NSString *)contextId
                                          modelId:(NSString *)modelId
                                     tokenArrays:(NSArray<NSArray<NSNumber *> *> *)tokenArrays) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto ci = g_contexts.find([contextId UTF8String]);
  auto mi = g_models.find([modelId UTF8String]);
  if (ci == g_contexts.end() || mi == g_models.end()) {
    @throw [NSException exceptionWithName:@"E_INVALID_HANDLE"
                                   reason:@"Invalid context or model ID for embedBatch()"
                                 userInfo:nil];
  }

  llama_context *ctx = ci->second;
  int n_seqs = (int)tokenArrays.count;
  int n_embd = llama_model_n_embd(mi->second);

  // Calculate total tokens
  int total_tokens = 0;
  for (NSArray *toks in tokenArrays) total_tokens += (int)toks.count;

  llama_batch batch = llama_batch_init(total_tokens, 0, n_seqs);
  int pos = 0;
  for (int seq = 0; seq < n_seqs; seq++) {
    NSArray *toks = tokenArrays[seq];
    for (int i = 0; i < (int)toks.count; i++) {
      llama_batch_add(batch, [toks[i] intValue], i, {seq}, i == (int)toks.count - 1);
    }
  }

  if (llama_encode(ctx, batch) != 0) {
    llama_batch_free(batch);
    return @[];
  }
  llama_batch_free(batch);

  NSMutableArray *results = [NSMutableArray arrayWithCapacity:n_seqs];
  for (int seq = 0; seq < n_seqs; seq++) {
    const float *embd = llama_get_embeddings_seq(ctx, seq);
    if (!embd) {
      [results addObject:@[]];
      continue;
    }

    float norm = 0.0f;
    for (int i = 0; i < n_embd; i++) norm += embd[i] * embd[i];
    norm = sqrtf(norm);

    NSMutableArray *vec = [NSMutableArray arrayWithCapacity:n_embd];
    for (int i = 0; i < n_embd; i++) {
      [vec addObject:@(norm > 0.0f ? embd[i] / norm : 0.0f)];
    }
    [results addObject:vec];
  }
  return results;
}

// ── Batch inference ──────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(startBatch:(NSString *)modelId
               contextId:(NSString *)contextId
                 prompts:(NSArray<NSString *> *)prompts
                 options:(NSDictionary *)options) {
  std::string ctxIdStr = [contextId UTF8String];

  dispatch_async(inference_queue(), ^{
    llama_model *model;
    llama_context *ctx;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      auto mi = g_models.find([modelId UTF8String]);
      auto ci = g_contexts.find(ctxIdStr);
      if (mi == g_models.end() || ci == g_contexts.end()) {
        [self sendEventWithName:@"onBatchToken" body:@{
          @"contextId": contextId, @"done": @YES, @"error": @"Not found", @"seqIndex": @(-1)
        }];
        return;
      }
      model = mi->second;
      ctx = ci->second;
    }

    SamplerParams sp = parse_sampler_params(options);
    const llama_vocab *vocab = llama_model_get_vocab(model);
    int n_seqs = (int)prompts.count;

    // Tokenize all prompts
    std::vector<std::vector<llama_token>> all_tokens(n_seqs);
    int total_tokens = 0;
    for (int s = 0; s < n_seqs; s++) {
      const char *cprompt = [prompts[s] UTF8String];
      int plen = (int)strlen(cprompt);
      all_tokens[s].resize(plen + 16);
      int n = llama_tokenize(vocab, cprompt, plen, all_tokens[s].data(), (int)all_tokens[s].size(), true, true);
      if (n < 0) {
        all_tokens[s].resize(-n);
        n = llama_tokenize(vocab, cprompt, plen, all_tokens[s].data(), (int)all_tokens[s].size(), true, true);
      }
      all_tokens[s].resize(n);
      total_tokens += n;
    }

    // Eval all prompts
    llama_batch batch = llama_batch_init(total_tokens, 0, n_seqs);
    for (int s = 0; s < n_seqs; s++) {
      for (int i = 0; i < (int)all_tokens[s].size(); i++) {
        llama_batch_add(batch, all_tokens[s][i], i, {s}, i == (int)all_tokens[s].size() - 1);
      }
    }
    if (llama_decode(ctx, batch) != 0) {
      llama_batch_free(batch);
      [self sendEventWithName:@"onBatchToken" body:@{
        @"contextId": contextId, @"done": @YES, @"error": @"Decode failed", @"seqIndex": @(-1)
      }];
      return;
    }
    llama_batch_free(batch);

    // Sample per sequence
    std::vector<llama_sampler *> samplers(n_seqs);
    std::vector<bool> done(n_seqs, false);
    std::vector<int> positions(n_seqs);
    for (int s = 0; s < n_seqs; s++) {
      samplers[s] = create_sampler(sp, model);
      positions[s] = (int)all_tokens[s].size();
    }

    bool cancelled = false;
    for (int iter = 0; iter < sp.max_tokens; iter++) {
      if (is_cancelled(ctxIdStr)) { cancelled = true; break; }

      bool all_done = true;
      for (int s = 0; s < n_seqs; s++) {
        if (done[s]) continue;
        all_done = false;

        llama_token new_token = llama_sampler_sample(samplers[s], ctx, -1);
        if (llama_vocab_is_eog(vocab, new_token)) {
          done[s] = true;
          [self sendEventWithName:@"onBatchToken" body:@{
            @"contextId": contextId, @"seqIndex": @(s), @"done": @YES, @"finishReason": @"stop"
          }];
          continue;
        }

        std::string piece = token_to_piece(model, new_token);
        [self sendEventWithName:@"onBatchToken" body:@{
          @"contextId": contextId, @"seqIndex": @(s),
          @"token": [NSString stringWithUTF8String:piece.c_str()], @"done": @NO
        }];

        llama_batch single = llama_batch_init(1, 0, n_seqs);
        llama_batch_add(single, new_token, positions[s], {s}, true);
        positions[s]++;
        llama_decode(ctx, single);
        llama_batch_free(single);
      }
      if (all_done) break;
    }

    // Cleanup
    for (auto *s : samplers) llama_sampler_free(s);
    clear_cancel(ctxIdStr);

    // Mark any remaining sequences as done
    NSString *reason = cancelled ? @"cancelled" : @"length";
    for (int s = 0; s < n_seqs; s++) {
      if (!done[s]) {
        [self sendEventWithName:@"onBatchToken" body:@{
          @"contextId": contextId, @"seqIndex": @(s), @"done": @YES, @"finishReason": reason
        }];
      }
    }
  });
}

// ── Quantization ─────────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(quantize:(NSString *)inputPath
             outputPath:(NSString *)outputPath
                options:(NSDictionary *)options) {
  dispatch_async(inference_queue(), ^{
    llama_model_quantize_params params = llama_model_quantize_default_params();
    params.ftype = options[@"ftype"] ? [options[@"ftype"] intValue] : 15; // Q4_K_M default
    if (options[@"nthread"])                 params.nthread = [options[@"nthread"] intValue];
    if (options[@"allow_requantize"])        params.allow_requantize = [options[@"allow_requantize"] boolValue];
    if (options[@"quantize_output_tensor"])  params.quantize_output_tensor = [options[@"quantize_output_tensor"] boolValue];
    if (options[@"pure"])                    params.pure = [options[@"pure"] boolValue];

    uint32_t result = llama_model_quantize([inputPath UTF8String], [outputPath UTF8String], &params);

    NSString *error = (result != 0) ? [NSString stringWithFormat:@"Quantization failed with code %u", result] : nil;
    [self sendEventWithName:@"onQuantizeComplete" body:@{
      @"error": error ?: [NSNull null],
    }];
  });
}

// ── Logging ──────────────────────────────────────────────────────────────────

static void llm_log_callback(enum ggml_log_level level, const char *text, void * /*user_data*/) {
  if (!g_log_events_enabled.load(std::memory_order_relaxed)) return;
  if ((int)level < g_log_min_level.load(std::memory_order_relaxed)) return;

  LocalLLM *module = g_log_module;
  if (!module || !module->_hasListeners) return;

  [module sendEventWithName:@"onLog" body:@{
    @"level": @((int)level),
    @"text": @(text),
  }];
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(setLogLevel:(double)level) {
  g_log_min_level.store((int)level, std::memory_order_relaxed);
  return nil;
}

RCT_EXPORT_METHOD(enableLogEvents:(BOOL)enabled) {
  g_log_events_enabled.store(enabled, std::memory_order_relaxed);
  if (enabled) {
    g_log_module = self;
    llama_log_set(llm_log_callback, nullptr);
  } else {
    llama_log_set(nullptr, nullptr);
    g_log_module = nil;
  }
}

// ── Downloads ────────────────────────────────────────────────────────────────

RCT_EXPORT_METHOD(downloadModel:(NSString *)url destPath:(NSString *)destPath) {
  NSURL *nsUrl = [NSURL URLWithString:url];
  if (!nsUrl) return;

  _downloadDelegate.destPaths[url] = destPath;
  NSURLSessionDownloadTask *task = [_downloadSession downloadTaskWithURL:nsUrl];
  [task resume];
}

RCT_EXPORT_METHOD(cancelDownload:(NSString *)url) {
  [_downloadSession getTasksWithCompletionHandler:^(NSArray *dataTasks, NSArray *uploadTasks, NSArray *downloadTasks) {
    for (NSURLSessionDownloadTask *task in downloadTasks) {
      if ([task.originalRequest.URL.absoluteString isEqualToString:url]) {
        [task cancel];
      }
    }
  }];
}

// ── Device capabilities ──────────────────────────────────────────────────────

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getDeviceCapabilities) {
  NSProcessInfo *info = [NSProcessInfo processInfo];
  id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();

  uint64_t totalRAM = info.physicalMemory;
  uint64_t availableRAM = os_proc_available_memory();

  NSOperatingSystemVersion ver = info.operatingSystemVersion;
  NSString *iosVersion = [NSString stringWithFormat:@"%ld.%ld.%ld",
    (long)ver.majorVersion, (long)ver.minorVersion, (long)ver.patchVersion];

  // Detect Metal GPU family
  int metalFamily = 0;
  if (gpu) {
    if ([gpu supportsFamily:MTLGPUFamilyApple9]) metalFamily = 9;
    else if ([gpu supportsFamily:MTLGPUFamilyApple8]) metalFamily = 8;
    else if ([gpu supportsFamily:MTLGPUFamilyApple7]) metalFamily = 7;
    else if ([gpu supportsFamily:MTLGPUFamilyApple6]) metalFamily = 6;
    else if ([gpu supportsFamily:MTLGPUFamilyApple5]) metalFamily = 5;
    else if ([gpu supportsFamily:MTLGPUFamilyApple4]) metalFamily = 4;
  }

  int metalVersion = metalFamily >= 7 ? 3 : metalFamily >= 5 ? 2 : 1;

  return @{
    @"totalRAM":        @(totalRAM),
    @"availableRAM":    @(availableRAM),
    @"gpuName":         gpu ? gpu.name : @"unknown",
    @"metalFamily":     @(metalFamily),
    @"metalVersion":    @(metalVersion),
    @"iosVersion":      iosVersion,
    @"isLowPowerMode":  @(info.isLowPowerModeEnabled),
  };
}

RCT_EXPORT_BLOCKING_SYNCHRONOUS_METHOD(getModelStoragePath) {
  NSArray *paths = NSSearchPathForDirectoriesInDomains(NSApplicationSupportDirectory, NSUserDomainMask, YES);
  NSString *appSupport = paths.firstObject;
  NSString *llmDir = [appSupport stringByAppendingPathComponent:@"local-llm/models"];

  NSFileManager *fm = [NSFileManager defaultManager];
  if (![fm fileExistsAtPath:llmDir]) {
    [fm createDirectoryAtPath:llmDir withIntermediateDirectories:YES attributes:nil error:nil];
  }
  return llmDir;
}

@end
