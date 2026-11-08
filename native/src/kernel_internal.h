#pragma once

#include <cstdint>
#include <vector>

namespace g2 {

#pragma pack(push,1)
struct Q40 {
    uint16_t fp16_scale;
    uint8_t packed[16];
};
#pragma pack(pop)
static_assert(sizeof(Q40)==18,"GGML Q4_0 block must be 18 bytes");

struct Matrix {
    int rows=0;
    int cols=0;
    std::vector<Q40> weights;
};

struct QuantizedInput {
    std::vector<int8_t> x8;
    std::vector<float> scales;
};

void dotprod_gemv_rows(const Matrix& matrix,const QuantizedInput& activations,
                       float* output,int first,int last);

}  // namespace g2
