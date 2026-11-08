#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct AAssetManager;

namespace g2 {

// This function performs an actual Vulkan compute dispatch on the phone.
// All upload/setup overhead is separated from per-dispatch wall-clock timing.
std::string vulkan_benchmark_json(
    const std::vector<uint32_t>& packed_q40,
    const std::vector<float>& activations,
    const std::vector<float>& reference,
    int rows, int cols, int repetitions,
    AAssetManager* assets);

}  // namespace g2
