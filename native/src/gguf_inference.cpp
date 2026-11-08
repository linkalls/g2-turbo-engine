// G2 Turbo Engine v0.5 — real GGUF model inference benchmark on Android.
//
// Separate metrics:
//   model_load_ms      : read/model-map + backend initialization
//   context_init_ms    : KV cache, compute graph context creation
//   tokenize_ms        : PFML3 prompt tokenization
//   prefill_eval_ms    : llama_decode() prompt batches, excluding tokenization
//   decode_eval_ms     : only llama_decode() for generated tokens (no sampler)
//   decode_tokens_eval : number of generated token evaluations
//   decode_tps         : decode_tokens_eval * 1000 / decode_eval_ms
//
// The first sampled token comes from the last prompt logits. It does not
// require a separate decode evaluation and is excluded from decode_tps.
// This avoids claiming an artificially inflated decode rate.
//
// Model weights are loaded ONCE and shared by contexts for 1/2/4 threads.
// This is the llama.cpp ARM DOTPROD CPU backend, not the separate standalone
// v0.4 G2 custom kernel. Never label the result as G2 hybrid inference.
#include <jni.h>
#include <llama.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        Clock::now() - start).count();
}

std::string quote_json(const std::string& src) {
    std::ostringstream out;
    out << '"';
    for (unsigned char c : src) {
        switch (c) {
            case '"': out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 32) {
                    const char* hex = "0123456789abcdef";
                    out << "\\u00" << hex[c >> 4] << hex[c & 15];
                } else {
                    out << static_cast<char>(c);
                }
        }
    }
    out << '"';
    return out.str();
}

struct Progress {
    JNIEnv* env;
    jobject activity;
    jmethodID on_progress;

    void report(const std::string& message) {
        if (!on_progress) return;
        jstring line = env->NewStringUTF(message.c_str());
        if (!line) return;
        env->CallVoidMethod(activity, on_progress, line);
        env->DeleteLocalRef(line);
        if (env->ExceptionCheck()) {
            env->ExceptionClear(); // UI callback must not kill native compute.
        }
    }
};

struct ModelDeleter {
    void operator()(llama_model* model) const {
        if (model) llama_model_free(model);
    }
};
struct ContextDeleter {
    void operator()(llama_context* ctx) const {
        if (ctx) llama_free(ctx);
    }
};
struct SamplerDeleter {
    void operator()(llama_sampler* s) const {
        if (s) llama_sampler_free(s);
    }
};

std::string token_piece(const llama_vocab* vocab, llama_token token) {
    char bytes[512];
    int result = llama_token_to_piece(
        vocab, token, bytes, sizeof(bytes), 0, false);
    if (result >= 0) {
        return std::string(bytes, size_t(result));
    }
    const int needed = -result;
    if (needed <= 0 || needed > 65536) return {};
    std::vector<char> buffer(static_cast<size_t>(needed));
    result = llama_token_to_piece(
        vocab, token, buffer.data(), needed, 0, false);
    if (result < 0) return {};
    return std::string(buffer.data(), size_t(result));
}

struct RunStats {
    int threads = 0;
    int prompt_tokens = 0;
    int output_tokens = 0;
    int decode_evaluations = 0;
    double context_init_ms = 0;
    double tokenization_ms = 0;
    double prefill_eval_ms = 0;
    double decode_eval_ms = 0;
    double sampler_ms = 0;
    double total_inference_ms = 0;
    std::string text;
};

RunStats perform_inference(llama_model* model,
                           const std::string& prompt,
                           int n_predict, int threads,
                           Progress& progress) {
    RunStats result;
    result.threads = threads;
    progress.report("CPU " + std::to_string(threads) +
                    "スレッド: KVキャッシュを準備中");

    const auto context_start = Clock::now();
    llama_context_params params = llama_context_default_params();
    params.n_ctx = 2048;
    params.n_batch = 256;
    params.n_ubatch = 256;
    params.n_threads = threads;
    params.n_threads_batch = threads;
    params.no_perf = false;
    params.offload_kqv = false;
    std::unique_ptr<llama_context, ContextDeleter> ctx(
        llama_init_from_model(model, params));
    if (!ctx) throw std::runtime_error("llama_init_from_model failed (RAM/KV?)");
    result.context_init_ms = elapsed_ms(context_start);

    // The model's GGUF tokenizer understands PFML3 special tokens.
    const llama_vocab* vocab = llama_model_get_vocab(model);
    const auto tokenize_start = Clock::now();
    int capacity = 4096;
    std::vector<llama_token> tokens(static_cast<size_t>(capacity));
    int count = llama_tokenize(vocab, prompt.data(),
                               static_cast<int32_t>(prompt.size()),
                               tokens.data(), capacity, true, true);
    if (count < 0) {
        if (count == INT32_MIN || -count > 8192)
            throw std::runtime_error("Prompt too large to tokenize");
        capacity = -count;
        tokens.resize(static_cast<size_t>(capacity));
        count = llama_tokenize(vocab, prompt.data(),
                               static_cast<int32_t>(prompt.size()),
                               tokens.data(), capacity, true, true);
    }
    if (count <= 0)
        throw std::runtime_error("llama_tokenize produced no prompt tokens");
    tokens.resize(static_cast<size_t>(count));
    result.tokenization_ms = elapsed_ms(tokenize_start);
    result.prompt_tokens = count;
    if (count + n_predict >= static_cast<int>(llama_n_ctx(ctx.get())))
        throw std::runtime_error("Prompt + generation exceeds 2048 token context");

    progress.report("CPU " + std::to_string(threads) +
                    "スレッド: Prefill " + std::to_string(count) + " tokens");

    const auto inference_start = Clock::now();
    for (int offset = 0; offset < count; offset += 256) {
        const int take = std::min(256, count - offset);
        llama_batch batch = llama_batch_get_one(tokens.data() + offset, take);
        const auto tick = Clock::now();
        const int status = llama_decode(ctx.get(), batch);
        result.prefill_eval_ms += elapsed_ms(tick);
        if (status != 0)
            throw std::runtime_error("Prefill llama_decode status=" +
                                     std::to_string(status));
    }

    std::unique_ptr<llama_sampler, SamplerDeleter> sampler(
        llama_sampler_init_greedy());
    if (!sampler) throw std::runtime_error("Could not allocate greedy sampler");

    progress.report("CPU " + std::to_string(threads) +
                    "スレッド: Decode開始");
    for (int step = 0; step < n_predict; ++step) {
        const auto sample_start = Clock::now();
        const llama_token next = llama_sampler_sample(
            sampler.get(), ctx.get(), -1);
        result.sampler_ms += elapsed_ms(sample_start);
        if (llama_vocab_is_eog(vocab, next)) break;

        const std::string piece = token_piece(vocab, next);
        // PFML3 ends its current output when it emits the next tag.
        if (piece.find("<|plamo:tag|>") != std::string::npos) break;

        result.text += piece;
        ++result.output_tokens;
        if (step + 1 >= n_predict) break;

        llama_token current = next;
        llama_batch batch = llama_batch_get_one(&current, 1);
        const auto decode_start = Clock::now();
        const int status = llama_decode(ctx.get(), batch);
        result.decode_eval_ms += elapsed_ms(decode_start);
        if (status != 0)
            throw std::runtime_error("Decode llama_decode status=" +
                                     std::to_string(status));
        ++result.decode_evaluations;

        if ((step + 1) % 16 == 0)
            progress.report("CPU " + std::to_string(threads) +
                            "スレッド: Decode " + std::to_string(step + 1) +
                            "/" + std::to_string(n_predict) + " tokens");
    }
    result.total_inference_ms = elapsed_ms(inference_start);
    return result;
}

std::string run_model(const std::string& path,
                      const std::string& prompt,
                      int n_predict, Progress& progress) {
    if (path.empty() || prompt.empty())
        throw std::runtime_error("Missing GGUF path or prompt");
    if (n_predict < 1 || n_predict > 512)
        throw std::runtime_error("n_predict must be 1 to 512");
    static std::once_flag backend_once;
    std::call_once(backend_once, [] { llama_backend_init(); });

    progress.report("実際のGGUF重みをロード中（RAM使用量に注意）");
    const auto load_start = Clock::now();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.load_mode = LLAMA_LOAD_MODE_MMAP;
    std::unique_ptr<llama_model, ModelDeleter> model(
        llama_model_load_from_file(path.c_str(), model_params));
    if (!model)
        throw std::runtime_error("GGUFの読み込みに失敗。対応形式・空きRAMを確認");
    const double model_load_ms = elapsed_ms(load_start);

    char model_desc[256]{};
    llama_model_desc(model.get(), model_desc, sizeof(model_desc));

    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\"engine\":\"G2 Turbo Engine v0.5 real GGUF\","
        << "\"backend\":\"llama.cpp ARM CPU, custom G2 kernels not connected\","
        << "\"llama_revision\":\"e14a779719652200cec7ad58dfa4820edc347182\","
        << "\"model_description\":" << quote_json(model_desc) << ","
        << "\"model_weight_bytes\":" << llama_model_size(model.get()) << ","
        << "\"model_layers\":" << llama_model_n_layer(model.get()) << ","
        << "\"model_load_ms\":" << model_load_ms << ","
        << "\"requested_output_tokens\":" << n_predict << ","
        << "\"context_tokens\":2048,"
        << "\"runs\":[";
    for (int i = 0; i < 3; ++i) {
        const int threads[] = {1, 2, 4};
        if (i) out << ",";
        try {
            RunStats st = perform_inference(
                model.get(), prompt, n_predict, threads[i], progress);
            out << "{\"threads\":" << threads[i]
                << ",\"status\":\"ok\","
                << "\"prompt_tokens\":" << st.prompt_tokens << ","
                << "\"output_tokens_generated\":" << st.output_tokens << ","
                << "\"decode_tokens_evaluated\":" << st.decode_evaluations << ","
                << "\"context_init_ms\":" << st.context_init_ms << ","
                << "\"tokenization_ms\":" << st.tokenization_ms << ","
                << "\"prefill_eval_ms\":" << st.prefill_eval_ms << ","
                << "\"prefill_tokens_per_s\":"
                << (st.prefill_eval_ms > 0 ?
                    1000.0 * st.prompt_tokens / st.prefill_eval_ms : 0) << ","
                << "\"decode_eval_ms\":" << st.decode_eval_ms << ","
                << "\"decode_tokens_per_s\":"
                << (st.decode_eval_ms > 0 ?
                    1000.0 * st.decode_evaluations / st.decode_eval_ms : 0) << ","
                << "\"sampler_ms\":" << st.sampler_ms << ","
                << "\"total_inference_ms\":" << st.total_inference_ms << ","
                << "\"generated_text\":" << quote_json(st.text)
                << "}";
        } catch (const std::exception& error) {
            out << "{\"threads\":" << threads[i]
                << ",\"status\":\"error\","
                << "\"message\":" << quote_json(error.what()) << "}";
        }
    }
    out << "]}";
    progress.report("実GGUFの全3スレッド設定の計測が完了");
    return out.str();
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_dev_poteto_g2turbo_ModelBenchmarkActivity_nativeRunGGUF(
        JNIEnv* env, jobject activity, jstring model_path,
        jstring prompt, jint n_predict) {
    const char* path_chars = env->GetStringUTFChars(model_path, nullptr);
    const char* prompt_chars = env->GetStringUTFChars(prompt, nullptr);
    const std::string path = path_chars ? path_chars : "";
    const std::string prompt_text = prompt_chars ? prompt_chars : "";
    if (path_chars) env->ReleaseStringUTFChars(model_path, path_chars);
    if (prompt_chars) env->ReleaseStringUTFChars(prompt, prompt_chars);

    const jclass activity_class = env->GetObjectClass(activity);
    const jmethodID on_progress = env->GetMethodID(
        activity_class, "onNativeProgress", "(Ljava/lang/String;)V");
    Progress progress{env, activity, on_progress};
    try {
        const std::string result = run_model(
            path, prompt_text, int(n_predict), progress);
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& error) {
        const std::string result =
            "{\"status\":\"error\",\"message\":" +
            quote_json(error.what()) + "}";
        return env->NewStringUTF(result.c_str());
    }
}
