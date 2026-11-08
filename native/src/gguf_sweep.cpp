// G2 Turbo Engine v0.6 — real GGUF translation length/output sweep.
//
// This benchmark always executes the actual model graph through llama.cpp.
// The custom standalone G2 and KleidiAI microbench kernels are NOT connected.
//
// Each testcase is recorded immediately via JNI callback. Long runs can be
// cancelled between prompt chunks or autoregressive decoding steps.
// Prompt targets are requested token counts: actual model token counts are
// measured and retained separately for scientific reproducibility.
#include <jni.h>
#include <llama.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
std::atomic_bool cancelled{false};

double millis(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}

std::string escape(const std::string& source) {
    static const char* hex = "0123456789abcdef";
    std::string result = "\"";
    for (unsigned char c : source) {
        if (c == '"') result += "\\\"";
        else if (c == '\\') result += "\\\\";
        else if (c == '\n') result += "\\n";
        else if (c == '\r') result += "\\r";
        else if (c == '\t') result += "\\t";
        else if (c < 0x20) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else result += static_cast<char>(c);
    }
    return result + "\"";
}

void check_cancel() {
    if (cancelled.load()) throw std::runtime_error("CANCELLED_BY_USER");
}

struct Callback {
    JNIEnv* env;
    jobject activity;
    jmethodID progressMethod;
    jmethodID caseMethod;
    void call(jmethodID method, const std::string& str) const {
        if (!method) return;
        jstring text = env->NewStringUTF(str.c_str());
        if (!text) return;
        env->CallVoidMethod(activity, method, text);
        env->DeleteLocalRef(text);
        if (env->ExceptionCheck()) env->ExceptionClear();
    }
    void progress(const std::string& str) const { call(progressMethod,str); }
    void result(const std::string& str) const { call(caseMethod,str); }
};

struct ModelFree { void operator()(llama_model* ptr) const {
    if (ptr) llama_model_free(ptr);
}};
struct ContextFree { void operator()(llama_context* ptr) const {
    if (ptr) llama_free(ptr);
}};
struct SamplerFree { void operator()(llama_sampler* ptr) const {
    if (ptr) llama_sampler_free(ptr);
}};
using Model = std::unique_ptr<llama_model,ModelFree>;
using Context = std::unique_ptr<llama_context,ContextFree>;
using Sampler = std::unique_ptr<llama_sampler,SamplerFree>;

std::vector<llama_token> tokenize(const llama_vocab* vocab,
                                  const std::string& prompt) {
    int length = std::max(64,int(prompt.size()) + 32);
    std::vector<llama_token> tokens(static_cast<size_t>(length));
    int count = llama_tokenize(vocab, prompt.data(),
                               int32_t(prompt.size()), tokens.data(),
                               length, true, true);
    if (count < 0) {
        const int required = -count;
        if (required < 1 || required > 20000)
            throw std::runtime_error("Tokenizer prompt too long");
        tokens.resize(size_t(required));
        count = llama_tokenize(vocab, prompt.data(), int32_t(prompt.size()),
                               tokens.data(), required, true, true);
    }
    if (count < 1) throw std::runtime_error("No prompt tokens");
    tokens.resize(size_t(count));
    return tokens;
}

std::string prefix() {
    return "<|plamo:tag|>dataset<|plamo:msg|>translation"
           "<|plamo:tag|>input<|plamo:key|>lang<|plamo:val|>en"
           "<|plamo:msg|>";
}
std::string suffix() {
    return "<|plamo:tag|>output<|plamo:key|>lang<|plamo:val|>ja"
           "<|plamo:msg|>";
}

// Repeats the user's actual English sample until the requested TOKEN length.
// It cuts at a UTF-8 boundary and searches by tokenizer measurements.
// About ±1 token is expected; actual number is always reported.
std::vector<llama_token> make_tokens(const llama_vocab* vocab,
                                      const std::string& source,
                                      int target_tokens) {
    const std::string base = source.empty()
        ? "Modern language models translate text directly on local devices. "
        : source;
    std::string repeat = base;
    if (repeat.back() != ' ' && repeat.back() != '\n') repeat += "\n";
    std::string body;
    const auto base_tokens = tokenize(vocab, prefix() + repeat + suffix());
    const int effective = std::max(1,int(base_tokens.size()) - 15);
    const int repetitions = std::max(2,target_tokens/effective + 4);
    if (repeat.size()*size_t(repetitions) > 1000000)
        throw std::runtime_error("Input text too long to replicate");
    body.reserve(repeat.size()*size_t(repetitions));
    for (int i=0;i<repetitions;++i) body+=repeat;
    for (int attempt=0; attempt<8; ++attempt) {
        if (int(tokenize(vocab,prefix()+body+suffix()).size()) >= target_tokens)
            break;
        if (body.size()*2 > 500000)
            throw std::runtime_error("Cannot reach token target under 500KB input limit");
        body+=body;
    }
    const auto total = tokenize(vocab,prefix()+body+suffix());
    if (int(total.size()) < target_tokens)
        throw std::runtime_error("Could not reach requested token count");
    size_t left=0,right=body.size();
    while (right>left+1) {
        size_t mid=left+(right-left)/2;
        // Never split a multibyte UTF-8 codepoint.
        while (mid>0 && mid<body.size() &&
               (static_cast<unsigned char>(body[mid]) & 0xc0u)==0x80u) --mid;
        if (mid<=left) break;
        const int length=int(tokenize(
            vocab,prefix()+body.substr(0,mid)+suffix()).size());
        if (length<target_tokens) left=mid;
        else right=mid;
    }
    auto target=tokenize(vocab,prefix()+body.substr(0,right)+suffix());
    if (std::abs(int(target.size())-target_tokens)>3)
        throw std::runtime_error("Token target mismatch");
    return target;
}

struct Measured {
    int actual_tokens=0;
    int generated_tokens=0;
    int evaluated_tokens=0;
    int requested_output=0;
    int context_tokens=0;
    int threads_batch=0;
    int threads_decode=0;
    double context_ms=0;
    double prefill_ms=0;
    double decode_ms=0;
    double sampling_ms=0;
    double inference_ms=0;
    std::string generated;
};

std::string piece(const llama_vocab* vocab, llama_token token) {
    std::vector<char> buf(8192);
    int count=llama_token_to_piece(vocab,token,buf.data(),int(buf.size()),0,false);
    if (count<0) return "";
    return std::string(buf.data(),size_t(count));
}

Measured run_one(llama_model* model,
                 const std::vector<llama_token>& input,
                 int max_output,
                 int prefill_threads, int decode_threads,
                 const Callback& callback,
                 const std::string& case_name) {
    Measured data;
    data.actual_tokens=int(input.size());
    data.requested_output=max_output;
    data.threads_batch=prefill_threads;
    data.threads_decode=decode_threads;

    // Allocate as little KV cache as possible per case while respecting
    // llama.cpp's granularity. 16k+ contexts require more RAM on mobile.
    const int needed = int(input.size())+max_output+32;
    const int n_ctx = (needed+255)/256*256;
    if (n_ctx>16384) throw std::runtime_error(
        "Prompt + output exceeds the 16384-token experiment limit");
    data.context_tokens=n_ctx;

    const auto ctx_start=Clock::now();
    llama_context_params options=llama_context_default_params();
    options.n_ctx=uint32_t(n_ctx);
    options.n_batch=256;
    options.n_ubatch=256;
    options.n_threads=decode_threads;
    options.n_threads_batch=prefill_threads;
    options.no_perf=false;
    options.offload_kqv=false;
    Context ctx(llama_init_from_model(model,options));
    if (!ctx) throw std::runtime_error(
        "Cannot create context/KV buffers: insufficient RAM?");
    data.context_ms=millis(ctx_start);
    // Explicitly enable different thread counts for prefill and decode.
    llama_set_n_threads(ctx.get(),decode_threads,prefill_threads);
    const llama_vocab* vocab=llama_model_get_vocab(model);
    Sampler sampler(llama_sampler_init_greedy());
    if (!sampler) throw std::runtime_error("llama_sampler_init_greedy failed");

    const auto inference_start=Clock::now();
    for (int pos=0; pos<int(input.size()); pos+=256) {
        check_cancel();
        const int chunk=std::min(256,int(input.size())-pos);
        llama_batch batch=llama_batch_get_one(
            const_cast<llama_token*>(input.data())+pos,chunk);
        const auto start=Clock::now();
        const int rc=llama_decode(ctx.get(),batch);
        data.prefill_ms+=millis(start);
        if (rc!=0) throw std::runtime_error(
            "Prefill failed, llama_decode rc="+std::to_string(rc));
        if (pos==0 || (pos/256)%8==0)
            callback.progress(case_name + " Prefill "
                + std::to_string(std::min(pos+chunk,int(input.size())))
                + "/" + std::to_string(input.size()));
    }
    for (int token=0; token<max_output; ++token) {
        check_cancel();
        const auto sample_start=Clock::now();
        llama_token next=llama_sampler_sample(sampler.get(),ctx.get(),-1);
        data.sampling_ms+=millis(sample_start);
        if (llama_vocab_is_eog(vocab,next)) break;
        const std::string rendered=piece(vocab,next);
        if (rendered.find("<|plamo:tag|>")!=std::string::npos) break;
        data.generated+=rendered;
        ++data.generated_tokens;
        if (token+1>=max_output) break;
        llama_batch b=llama_batch_get_one(&next,1);
        const auto start=Clock::now();
        const int rc=llama_decode(ctx.get(),b);
        data.decode_ms+=millis(start);
        if (rc!=0) throw std::runtime_error(
            "Decode failed, llama_decode rc="+std::to_string(rc));
        ++data.evaluated_tokens;
        if ((token+1)%32==0)
            callback.progress(case_name + " Decode "+std::to_string(token+1)
                + "/" + std::to_string(max_output));
    }
    data.inference_ms=millis(inference_start);
    return data;
}

std::string serialized(const Measured& d, int target, const std::string& id,
                       const std::string& quantization) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6)
        << "{\"status\":\"ok\",\"case_id\":" << escape(id)
        << ",\"model_quantization\":" << escape(quantization)
        << ",\"target_prompt_tokens\":" << target
        << ",\"actual_prompt_tokens\":" << d.actual_tokens
        << ",\"max_output_tokens\":" << d.requested_output
        << ",\"output_tokens_generated\":" << d.generated_tokens
        << ",\"decode_tokens_evaluated\":" << d.evaluated_tokens
        << ",\"prefill_threads\":" << d.threads_batch
        << ",\"decode_threads\":" << d.threads_decode
        << ",\"context_tokens\":" << d.context_tokens
        << ",\"context_init_ms\":" << d.context_ms
        << ",\"prefill_eval_ms\":" << d.prefill_ms
        << ",\"prefill_tok_s\":" << (d.prefill_ms>0?
             d.actual_tokens*1000.0/d.prefill_ms:0.0)
        << ",\"decode_eval_ms\":" << d.decode_ms
        << ",\"decode_tok_s\":" << (d.decode_ms>0?
             d.evaluated_tokens*1000.0/d.decode_ms:0.0)
        << ",\"sampling_ms\":" << d.sampling_ms
        << ",\"inference_ms\":" << d.inference_ms
        << ",\"completed_at_cap\":"
        << (d.generated_tokens>=d.requested_output?"true":"false")
        << ",\"generated_text\":" << escape(d.generated) << "}";
    return out.str();
}

std::vector<int> read_array(JNIEnv* env,jintArray arr) {
    if (!arr) throw std::runtime_error("Missing benchmark preset array");
    const jsize n=env->GetArrayLength(arr);
    if (n<1 || n>12) throw std::runtime_error("Invalid preset count");
    std::vector<jint> temp(static_cast<size_t>(n));
    env->GetIntArrayRegion(arr,0,n,temp.data());
    return std::vector<int>(temp.begin(),temp.end());
}
std::string from_java(JNIEnv* env,jstring str) {
    if (!str) return {};
    const char* chars=env->GetStringUTFChars(str,nullptr);
    std::string result=chars?chars:"";
    if (chars) env->ReleaseStringUTFChars(str,chars);
    return result;
}

std::string execute(JNIEnv* env,jobject activity,
                    const std::string& path, const std::string& sample,
                    const std::vector<int>& lengths,
                    const std::vector<int>& outputs,
                    const std::vector<int>& modes) {
    const auto cls=env->GetObjectClass(activity);
    const Callback cb{
        env,activity,
        env->GetMethodID(cls,"onNativeProgress","(Ljava/lang/String;)V"),
        env->GetMethodID(cls,"onNativeCaseResult","(Ljava/lang/String;)V")
    };
    cancelled=false;
    static std::once_flag once;
    std::call_once(once,[]{llama_backend_init();});
    cb.progress("GGUFをロード中（" + std::to_string(lengths.size()*outputs.size()*modes.size())
                + "ケース）");

    llama_model_params mp=llama_model_default_params();
    mp.n_gpu_layers=0;
    mp.load_mode=LLAMA_LOAD_MODE_MMAP;
    const auto load_start=Clock::now();
    Model model(llama_model_load_from_file(path.c_str(),mp));
    if(!model) throw std::runtime_error("Cannot load GGUF file");
    const double load_ms=millis(load_start);
    const std::string quantization=
        llama_ftype_name(llama_model_ftype(model.get()));
    const llama_vocab* vocab=llama_model_get_vocab(model.get());
    const int training_ctx=llama_model_n_ctx_train(model.get());

    std::ostringstream meta;
    meta << std::fixed << std::setprecision(6)
         << "{\"engine\":\"G2 Turbo Engine v0.6 real GGUF sweep\","
         << "\"backend\":\"llama.cpp CPU armv8.2-a+dotprod (not G2 custom kernel)\","
         << "\"model_quantization\":" << escape(quantization)
         << ",\"model_weights_bytes\":" << llama_model_size(model.get())
         << ",\"model_layers\":" << llama_model_n_layer(model.get())
         << ",\"model_train_context\":" << training_ctx
         << ",\"model_load_ms\":" << load_ms
         << ",\"requested_cases\":" << lengths.size()*outputs.size()*modes.size()
         << ",\"status\":\"ok\"}";

    // Report model metadata separately so partial results still identify
    // the actual quantization (rather than relying on a filename label).
    cb.result("{\"kind\":\"model_meta\",\"data\":" + meta.str() + "}");

    int completed=0,failed=0;
    for (int target:lengths) {
        check_cancel();
        if (target<32 || target>12000)
            throw std::runtime_error("Prompt target range: 32..12000");
        cb.progress("Token count "+std::to_string(target)+": generating prompt");
        std::vector<llama_token> prompt=make_tokens(vocab,sample,target);
        for (int output:outputs) {
            if (output<1 || output>2048)
                throw std::runtime_error("Max output range: 1..2048");
            for (size_t i=0;i<modes.size();++i) {
                check_cancel();
                // Rotate profile order between cases to reduce fixed-order
                // thermal and DVFS bias toward the first CPU configuration.
                const size_t rotate=(size_t(target/100)+size_t(output/32))
                                     % modes.size();
                const int mode=modes[(i+rotate)%modes.size()];
                int prefill=2,decode=2;
                if(mode==0) prefill=decode=1;
                else if(mode==1) prefill=decode=2;
                else if(mode==2) prefill=decode=4;
                else if(mode==3) {prefill=4;decode=2;}
                else throw std::runtime_error("Invalid profile code");

                const std::string id=std::to_string(target)+"-"+std::to_string(output)
                         +"-"+std::to_string(prefill)+"x"+std::to_string(decode);
                const std::string info="["+std::to_string(completed+failed+1)+"/"
                         +std::to_string(lengths.size()*outputs.size()*modes.size())
                         +"] "+id;
                cb.progress(info+" コンテキスト準備");
                std::string record;
                try {
                    const Measured m=run_one(model.get(),prompt,output,prefill,decode,cb,info);
                    record=serialized(m,target,id,quantization);
                    ++completed;
                } catch(const std::exception& e) {
                    if (cancelled.load()) throw;
                    record="{\"status\":\"error\",\"case_id\":"+escape(id)
                         +",\"target_prompt_tokens\":"+std::to_string(target)
                         +",\"actual_prompt_tokens\":"+std::to_string(prompt.size())
                         +",\"max_output_tokens\":"+std::to_string(output)
                         +",\"prefill_threads\":"+std::to_string(prefill)
                         +",\"decode_threads\":"+std::to_string(decode)
                         +",\"message\":"+escape(e.what())+"}";
                    ++failed;
                }
                // The Java callback persists this case before proceeding.
                cb.result("{\"kind\":\"case\",\"data\":"+record+"}");
            }
        }
    }
    cb.progress("完了: "+std::to_string(completed)+" 成功、"
                +std::to_string(failed)+" 失敗");
    return "{\"status\":\"ok\",\"completed\":"+std::to_string(completed)
         +",\"failed\":"+std::to_string(failed)
         +",\"model_metadata\":"+meta.str()+"}";
}

} // namespace

extern "C" JNIEXPORT void JNICALL
Java_dev_poteto_g2turbo_SweepBenchmarkActivity_nativeCancelSweep(
        JNIEnv*,jobject) {
    cancelled.store(true);
}

extern "C" JNIEXPORT jstring JNICALL
Java_dev_poteto_g2turbo_SweepBenchmarkActivity_nativeRunSweep(
        JNIEnv* env,jobject activity,jstring jpath,jstring jsource,
        jintArray jlengths,jintArray joutputs,jintArray jmodes) {
    try {
        const std::string path=from_java(env,jpath);
        const std::string source=from_java(env,jsource);
        const std::vector<int> lengths=read_array(env,jlengths);
        const std::vector<int> outputs=read_array(env,joutputs);
        const std::vector<int> modes=read_array(env,jmodes);
        std::string result=execute(env,activity,path,source,lengths,outputs,modes);
        return env->NewStringUTF(result.c_str());
    } catch(const std::exception& e) {
        const std::string msg="{\"status\":\"error\",\"message\":"+escape(e.what())+"}";
        return env->NewStringUTF(msg.c_str());
    }
}
