// G2 Turbo Engine v0.4 — Arm KleidiAI Q4_0 comparison.
// Uses unmodified upstream C micro-kernels, pinned v1.9.0 (Apache-2.0).
//
// Important: KleidiAI expects GGML-style Q4_0 scale FIRST then 16 packed bytes.
// Matrix is N output rows x K features, LHS is one FP32 activation row.
// Pack RHS once, pack LHS per inference, run one dot-product GEMV.
// This reports both packing costs and kernel latency separately.
#include "kleidi_bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

#include "kai/kai_common.h"
#include "kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.h"
#include "kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.h"
#include "kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.h"

namespace g2 {
namespace {
using Clock = std::chrono::steady_clock;
template<class F> double time_ms(F f, int repetitions) {
    const auto start = Clock::now();
    for (int i = 0; i < repetitions; ++i) f();
    const auto end = Clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count()/repetitions;
}
}
std::string kleidi_benchmark_json(const uint8_t* q4_weights, int rows, int cols,
                                  const std::vector<float>& activations,
                                  const std::vector<float>& reference,
                                  int repetitions) {
    try {
        using namespace std;
        const size_t m=1, n=static_cast<size_t>(rows),
                     k=static_cast<size_t>(cols), bl=32;
        const auto get_mr=kai_get_mr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod;
        const auto get_nr=kai_get_nr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod;
        const auto get_kr=kai_get_kr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod;
        const auto get_sr=kai_get_sr_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod;
        const size_t mr=get_mr(), nr=get_nr(), kr=get_kr(), sr=get_sr();
        if (mr == 0 || nr == 0 || kr == 0 || sr == 0) throw std::runtime_error("Invalid KleidiAI block dimensions");
        const size_t rhs_size=kai_get_rhs_packed_size_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(n,k,nr,kr,bl);
        const size_t lhs_size=kai_get_lhs_packed_size_lhs_quant_pack_qsi8d32p_f32(m,k,bl,mr,kr,sr);
        vector<uint8_t> rhs(rhs_size), lhs(lhs_size);
        vector<float> output(n);
        kai_rhs_pack_qs4cxs1s0_param params{};
        params.lhs_zero_point=1;
        params.rhs_zero_point=8;
        const double rhs_pack_ms=time_ms([&] {
            kai_run_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0(
                1,n,k,nr,kr,sr,bl,q4_weights,nullptr,rhs.data(),0,&params);
        },1);

        const auto lhs_pack=[&] {
            kai_run_lhs_quant_pack_qsi8d32p_f32(m,k,bl,mr,kr,sr,0,
                activations.data(),k*sizeof(float),lhs.data());
        };
        const auto gemv=[&] {
            kai_run_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod(
                m,n,k,bl,lhs.data(),rhs.data(),output.data(),
                n*sizeof(float),sizeof(float),
                -std::numeric_limits<float>::max(),
                std::numeric_limits<float>::max());
        };
        lhs_pack();
        gemv();
        const double lhs_pack_ms=time_ms(lhs_pack,repetitions);
        const double kernel_ms=time_ms(gemv,repetitions);

        double e2=0, ref2=0, maxe=0;
        for (int row=0; row<rows; ++row) {
            const double delta=double(output[row])-double(reference[row]);
            e2+=delta*delta; ref2+=double(reference[row])*double(reference[row]);
            maxe=std::max(maxe,std::abs(delta));
        }
        const double nrmse=std::sqrt(e2/std::max(ref2,1e-30));
        const bool pass=std::isfinite(nrmse) && nrmse<0.05;
        std::ostringstream out;
        out<<std::fixed<<std::setprecision(6)
           <<"{\"status\":\""<<(pass?"ok":"incorrect")<<"\","
           <<"\"library\":\"Arm KleidiAI v1.9.0\","
           <<"\"kernel\":\"qsi8d32p1x8_qsi4c32p4x8_neon_dotprod\","
           <<"\"lhs_pack_ms\":"<<lhs_pack_ms<<","
           <<"\"rhs_pack_once_ms\":"<<rhs_pack_ms<<","
           <<"\"gemv_kernel_ms\":"<<kernel_ms<<","
           <<"\"total_steady_state_ms\":"<<(lhs_pack_ms+kernel_ms)<<","
           <<"\"normalized_rmse\":"<<nrmse<<","
           <<"\"max_absolute_error\":"<<maxe<<","
           <<"\"pass\":"<<(pass?"true":"false")<<"}";
        return out.str();
    } catch (const std::exception& e) {
        // Keep the entire v0.4 suite usable if KleidiAI alone fails.
        return std::string("{\"status\":\"error\",\"message\":\"")+e.what()+"\",\"pass\":false}";
    }
}
} // namespace g2
