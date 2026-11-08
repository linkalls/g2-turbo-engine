#pragma once

#include <string>

namespace g2 {
struct Matrix;
struct QuantizedInput;

std::string cpu_pool_benchmark_json(
    const Matrix& matrix, const QuantizedInput& quantized,
    const float* reference, int repetitions, double activation_quant_ms);
}
