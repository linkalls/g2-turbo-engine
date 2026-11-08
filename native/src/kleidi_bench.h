#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace g2 {
// Invokes Arm KleidiAI's upstream Q4_0 SDOT micro-kernel.
// The model RHS packing occurs once; reported compute time excludes it.
std::string kleidi_benchmark_json(const uint8_t* q4_weights, int rows, int cols,
                                  const std::vector<float>& activations,
                                  const std::vector<float>& reference,
                                  int repetitions);
}
