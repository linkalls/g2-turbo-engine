// G2 Turbo Engine / experimental Q4_0 matrix-vector kernel.
// The optimized path uses ARMv8.2 SDOT instructions on the Pixel 7 Pro.
// This is a microbenchmark, NOT an end-to-end PLaMo runtime.
//
// Standard GGML Q4_0 layout: one 16-bit FP16 scale and 16 packed bytes.
// In every byte, the low nibble maps to weights 0..15, and high to 16..31.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include "kernel_internal.h"
#include "cpu_pool.h"

struct AAssetManager;  // Forward declaration for Android-only benchmarking.

#ifdef __ANDROID__
# include "vulkan_bench.h"
# include "kleidi_bench.h"
# include <android/asset_manager_jni.h>
#endif

#if defined(__aarch64__) && defined(__ARM_FEATURE_DOTPROD)
#  include <arm_neon.h>
#  define G2_HAS_DOTPROD 1
#else
#  define G2_HAS_DOTPROD 0
#endif

#ifdef __ANDROID__
#  include <jni.h>
#else
#  include <iostream>
#endif

namespace g2 {

constexpr int kBlock = 32;

float half_to_float(uint16_t bits) {
    const uint32_t sign = uint32_t(bits & 0x8000u) << 16u;
    const uint32_t e = (bits >> 10u) & 0x1fu;
    const uint32_t f = bits & 0x3ffu;
    uint32_t out = 0;
    if (e == 0) {
        if (f == 0) {
            out = sign;
        } else {
            int exp = -14;
            uint32_t fraction = f;
            while (!(fraction & 0x400u)) {
                fraction <<= 1;
                --exp;
            }
            out = sign | (uint32_t(exp + 127) << 23u)
                       | ((fraction & 0x3ffu) << 13u);
        }
    } else if (e == 31) {
        out = sign | 0x7f800000u | (f << 13u);
    } else {
        out = sign | ((e + 112u) << 23u) | (f << 13u);
    }
    float value = 0;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

Matrix make_matrix(int rows, int cols) {
    if (rows <= 0 || cols <= 0 || cols % kBlock != 0)
        throw std::invalid_argument("cols must be a positive multiple of 32");
    Matrix matrix{rows, cols, {}};
    matrix.weights.resize(size_t(rows) * size_t(cols / kBlock));
    std::mt19937 generator(0x71337u);
    std::uniform_int_distribution<int> nibble(0, 15);
    // Vary the FP16 scale between blocks; avoid conflating a constant with a real model.
    const uint16_t scales[] = {0x2800u, 0x2c00u, 0x3000u, 0x3100u};
    for (auto& block : matrix.weights) {
        block.fp16_scale = scales[generator() % 4u];
        for (uint8_t& q : block.packed)
            q = uint8_t(nibble(generator) | (nibble(generator) << 4));
    }
    return matrix;
}

std::vector<float> make_activations(int cols) {
    std::mt19937 generator(0x71000u);
    std::normal_distribution<float> normal(0.0f, 0.75f);
    std::vector<float> values(static_cast<size_t>(cols));
    for (float& x : values) x = normal(generator);
    return values;
}

// Golden reference: Q4_0 unpack and float accumulation.
// This is intentionally scalar so we have an independent correctness baseline.
void reference_gemv(const Matrix& m, const float* x, float* out) {
    const int blocks = m.cols / kBlock;
    for (int r = 0; r < m.rows; ++r) {
        const Q40* row = m.weights.data() + size_t(r) * blocks;
        float result = 0.0f;
        for (int b = 0; b < blocks; ++b) {
            const Q40& block = row[b];
            const float scale = half_to_float(block.fp16_scale);
            const float* input = x + b * kBlock;
            float dot = 0.0f;
            for (int j = 0; j < 16; ++j) {
                const uint8_t byte = block.packed[j];
                dot += float(int(byte & 15u) - 8) * input[j];
                dot += float(int(byte >> 4u) - 8) * input[j + 16];
            }
            result += scale * dot;
        }
        out[r] = result;
    }
}

QuantizedInput quantize(const float* x, int cols) {
    QuantizedInput q;
    q.x8.resize(size_t(cols));
    q.scales.resize(size_t(cols / kBlock));
    for (int b = 0; b < cols / kBlock; ++b) {
        const float* src = x + b * kBlock;
        float largest = 0;
        for (int j = 0; j < kBlock; ++j)
            largest = std::max(largest, std::fabs(src[j]));
        const float step = largest > 0.0f ? largest / 127.0f : 1.0f;
        q.scales[b] = step;
        for (int j = 0; j < kBlock; ++j) {
            int value = static_cast<int>(std::round(src[j] / step));
            q.x8[size_t(b * kBlock + j)] =
                static_cast<int8_t>(std::max(-127, std::min(127, value)));
        }
    }
    return q;
}

// Optimized path: 4-bit nibble unpacking to signed int8, followed by SDOT.
// The input vector is quantized once and reused across rows.
void dotprod_gemv_rows(const Matrix& m, const QuantizedInput& x, float* out, int first, int last) {
    const int blocks = m.cols / kBlock;
    for (int r = first; r < last; ++r) {
        const Q40* row = m.weights.data() + size_t(r) * blocks;
        float result = 0;
        for (int b = 0; b < blocks; ++b) {
            const Q40& block = row[b];
            const int8_t* input = x.x8.data() + b * kBlock;
            int32_t dot = 0;
#if G2_HAS_DOTPROD
            const uint8x16_t packed = vld1q_u8(block.packed);
            const uint8x16_t mask = vdupq_n_u8(15u);
            const uint8x16_t bias = vdupq_n_u8(8u);
            const int8x16_t low = vreinterpretq_s8_u8(
                vsubq_u8(vandq_u8(packed, mask), bias));
            const int8x16_t high = vreinterpretq_s8_u8(
                vsubq_u8(vshrq_n_u8(packed, 4), bias));
            int32x4_t accumulator = vdupq_n_s32(0);
            accumulator = vdotq_s32(accumulator, low, vld1q_s8(input));
            accumulator = vdotq_s32(accumulator, high, vld1q_s8(input + 16));
            dot = vaddvq_s32(accumulator);
#else
            for (int j = 0; j < 16; ++j) {
                const uint8_t byte = block.packed[j];
                dot += (int(byte & 15u) - 8) * int(input[j]);
                dot += (int(byte >> 4u) - 8) * int(input[j + 16]);
            }
#endif
            result += float(dot) * half_to_float(block.fp16_scale) * x.scales[b];
        }
        out[r] = result;
    }
}

// Partition matrix rows between workers. Each worker writes disjoint outputs.
void dotprod_gemv_parallel(const Matrix& m, const QuantizedInput& x,
                           float* out, int threads) {
    const int count = std::max(1, std::min(threads, m.rows));
    if (count == 1) {
        dotprod_gemv_rows(m, x, out, 0, m.rows);
        return;
    }
    // This benchmark includes per-inference thread startup and join costs.
    // A persistent worker pool may reduce these costs in a real LLM runtime.
    std::vector<std::thread> workers;
    workers.reserve(size_t(count-1));
    for (int i=1; i<count; ++i) {
        const int from = int(int64_t(m.rows)*i/count);
        const int to = int(int64_t(m.rows)*(i+1)/count);
        workers.emplace_back([&m,&x,out,from,to] {
            dotprod_gemv_rows(m,x,out,from,to);
        });
    }
    dotprod_gemv_rows(m,x,out,0,m.rows/count);
    for (auto& worker : workers) worker.join();
}
void dotprod_gemv(const Matrix& m, const QuantizedInput& x, float* out) {
    dotprod_gemv_parallel(m,x,out,1);
}

// Repacks GGUF Q4_0 18-byte blocks into 5x32-bit words for the planned
// Vulkan kernel: first word holds FP16 scale; remaining four hold packed bytes.
// The Android GPU path dispatches this packed layout through Vulkan.
std::vector<uint32_t> repack_for_gpu(const Matrix& m) {
    std::vector<uint32_t> result(m.weights.size() * 5u);
    for (size_t i = 0; i < m.weights.size(); ++i) {
        result[i * 5u] = m.weights[i].fp16_scale;
        for (size_t word = 0; word < 4; ++word) {
            const uint8_t* src = m.weights[i].packed + word * 4u;
            result[i * 5u + 1u + word] =
                uint32_t(src[0]) | (uint32_t(src[1]) << 8u) |
                (uint32_t(src[2]) << 16u) | (uint32_t(src[3]) << 24u);
        }
    }
    return result;
}

template<class F>
double milliseconds(F function, int n) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i) function();
    const auto end = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count() / n;
}

std::string bench(int rows, int cols, int repetitions,
                  AAssetManager* assets = nullptr) {
#ifndef __ANDROID__
    (void)assets;
#endif
    if (rows < 1 || rows > 8192 || cols < 32 || cols > 16384 ||
        cols % 32 != 0 || repetitions < 1 || repetitions > 100)
        throw std::invalid_argument("invalid benchmark dimensions");

    const Matrix matrix = make_matrix(rows, cols);
    const std::vector<float> x = make_activations(cols);
    const auto gpu = repack_for_gpu(matrix);

    for (size_t i = 0; i < matrix.weights.size(); ++i) {
        if (gpu[i * 5u] != matrix.weights[i].fp16_scale)
            throw std::runtime_error("GPU packing scale mismatch");
        for (size_t j = 0; j < 16; ++j) {
            const uint32_t word = gpu[i * 5u + 1u + j / 4u];
            if (uint8_t(word >> (8u * (j % 4u))) != matrix.weights[i].packed[j])
                throw std::runtime_error("GPU packing byte mismatch");
        }
    }

    const QuantizedInput quantized = quantize(x.data(), cols);
    std::vector<float> reference(static_cast<size_t>(rows));
    std::vector<float> optimized(static_cast<size_t>(rows));

    reference_gemv(matrix, x.data(), reference.data());
    dotprod_gemv(matrix, quantized, optimized.data());

    double error_squared = 0, reference_squared = 0;
    double max_error = 0, reference_checksum = 0, optimized_checksum = 0;
    for (int i = 0; i < rows; ++i) {
        const double difference = double(reference[i]) - double(optimized[i]);
        error_squared += difference * difference;
        reference_squared += double(reference[i]) * double(reference[i]);
        max_error = std::max(max_error, std::fabs(difference));
        reference_checksum += reference[i];
        optimized_checksum += optimized[i];
    }
    const double nrmse = std::sqrt(error_squared / std::max(reference_squared, 1e-30));

    const double quant_ms = milliseconds(
        [&] { auto temp = quantize(x.data(), cols);
              // Keep the quantization output observable for release optimizers.
              volatile int observe = temp.x8[0];
              (void)observe; }, repetitions);

    reference_gemv(matrix, x.data(), reference.data());
    dotprod_gemv(matrix, quantized, optimized.data());
    const double reference_ms = milliseconds(
        [&] { reference_gemv(matrix, x.data(), reference.data()); }, repetitions);
    const double dotprod_ms = milliseconds(
        [&] { dotprod_gemv(matrix, quantized, optimized.data()); }, repetitions);

    const bool pass = std::isfinite(nrmse) && nrmse < 0.05;
    const std::string pool_results = cpu_pool_benchmark_json(
        matrix, quantized, reference.data(), repetitions, quant_ms);

    // Compare CPU 1/2/4 threads using IDENTICAL weights and activations.
    // Quantization is reused within one inference; its cost is included.
    std::ostringstream thread_results;
    thread_results << std::fixed << std::setprecision(6) << "[";
    for (int threads : {1,2,4}) {
        if (threads > 1) thread_results << ",";
        const double elapsed = milliseconds([&] {
            dotprod_gemv_parallel(matrix, quantized, optimized.data(), threads);
        }, repetitions);
        double diff2 = 0;
        for (int row=0;row<rows;++row) {
            const double delta = optimized[row]-reference[row];
            diff2 += delta*delta;
        }
        const double thread_nrmse = std::sqrt(
            diff2/std::max(reference_squared,1e-30));
        thread_results << "{\"threads\":" << threads
                       << ",\"kernel_wall_ms\":" << elapsed
                       << ",\"total_with_activation_quant_ms\":" << (elapsed + quant_ms)
                       << ",\"nrmse\":" << thread_nrmse
                       << ",\"pass\":" << (std::isfinite(thread_nrmse) && thread_nrmse < 0.05 ? "true" : "false")
                       << "}";
    }
    thread_results << "]";
    std::ostringstream json;
    json << std::fixed << std::setprecision(6)
         << "{\n"
         << "  \"engine\": \"G2 Turbo Engine v0.4 Q4_0 GEMV\",\n"
         << "  \"full_llm_inference\": false,\n"
         #ifdef __ANDROID__
         << "  \"gpu_dispatch_implemented\": true,\n"
#else
         << "  \"gpu_dispatch_implemented\": false,\n"
#endif
         << "  \"arm_dotprod_compiled\": " << (G2_HAS_DOTPROD ? "true" : "false") << ",\n"
         << "  \"rows\": " << rows << ",\n"
         << "  \"cols\": " << cols << ",\n"
         << "  \"repetitions\": " << repetitions << ",\n"
         << "  \"reference_ms\": " << reference_ms << ",\n"
         << "  \"dotprod_ms\": " << dotprod_ms << ",\n"
         << "  \"activation_quantization_ms\": " << quant_ms << ",\n"
         << "  \"speedup_including_quantization\": "
         << (reference_ms / std::max(1e-12, dotprod_ms + quant_ms)) << ",\n"
         << "  \"normalized_rmse\": " << nrmse << ",\n"
         << "  \"max_absolute_error\": " << max_error << ",\n"
         << "  \"reference_checksum\": " << reference_checksum << ",\n"
         << "  \"optimized_checksum\": " << optimized_checksum << ",\n"
         << "  \"cpu_thread_benchmarks\": " << thread_results.str() << ",\n"
         << "  \"cpu_persistent_pool\": " << pool_results << ",\n"
#ifdef __ANDROID__
         << "  \"kleidiai_integrated\": true,\n"
         << "  \"kleidiai\": " << kleidi_benchmark_json(
             reinterpret_cast<const uint8_t*>(matrix.weights.data()),
             rows, cols, x, reference, repetitions) << ",\n"
#else
         << "  \"kleidiai_integrated\": false,\n"
#endif
         << "  \"pass\": " << (pass ? "true" : "false")
#ifdef __ANDROID__
         << ",\n  \"vulkan\": "
         << vulkan_benchmark_json(gpu, x, reference, rows, cols,
                                   repetitions, assets)
#endif
         << "\n}\n";
    return json.str();
}

} // namespace g2

#ifdef __ANDROID__
extern "C" JNIEXPORT jstring JNICALL
Java_dev_poteto_g2turbo_MainActivity_nativeBenchmark(
    JNIEnv* env, jobject, jint rows, jint cols, jint repetitions,
    jobject asset_manager) {
    try {
        const std::string result = g2::bench(rows, cols, repetitions,
                         AAssetManager_fromJava(env, asset_manager));
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& error) {
        const std::string message = std::string("Benchmark error: ") + error.what();
        return env->NewStringUTF(message.c_str());
    }
}
#else
int main(int argc, char** argv) {
    try {
        const int rows = argc > 1 ? std::stoi(argv[1]) : 768;
        const int cols = argc > 2 ? std::stoi(argv[2]) : 1536;
        const int repetitions = argc > 3 ? std::stoi(argv[3]) : 12;
        const std::string result = g2::bench(rows, cols, repetitions);
        std::cout << result;
        return result.find("\"pass\": true") != std::string::npos ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
#endif
