cmake_minimum_required(VERSION 3.22)
project(g2_turbo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ANDROID)
    # Pin official Arm KleidiAI and compile only the three kernels needed
    # for a fair Q4_0 GEMV comparison. Other Arm extensions are not required.
    set(KLEIDI_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kleidiai")
    if(NOT EXISTS "${KLEIDI_SOURCE_DIR}/kai/kai_common.h")
        message(FATAL_ERROR "KleidiAI v1.9.0 missing. Run scripts/fetch-kleidiai.sh")
    endif()
    add_library(g2kleidi STATIC
        "${KLEIDI_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_lhs_quant_pack_qsi8d32p_f32.c"
        "${KLEIDI_SOURCE_DIR}/kai/ukernels/matmul/pack/kai_rhs_pack_nxk_qsi4c32pscalef16_qsu4c32s16s0.c"
        "${KLEIDI_SOURCE_DIR}/kai/ukernels/matmul/matmul_clamp_f32_qsi8d32p_qsi4c32p/kai_matmul_clamp_f32_qsi8d32p1x8_qsi4c32p4x8_1x4x32_neon_dotprod.c"
    )
    target_include_directories(g2kleidi PUBLIC "${KLEIDI_SOURCE_DIR}")
    target_compile_options(g2kleidi PRIVATE -O3 -march=armv8.2-a+dotprod)
    add_library(g2turbo SHARED
        src/kernel_bench.cpp src/cpu_pool.cpp src/vulkan_bench.cpp src/kleidi_bench.cpp)
    target_compile_options(g2turbo PRIVATE -O3 -ffp-contract=fast -march=armv8.2-a+dotprod)
    target_link_libraries(g2turbo PRIVATE g2kleidi log android vulkan)
else()
    add_executable(g2-kernel-bench src/kernel_bench.cpp src/cpu_pool.cpp)
    target_compile_options(g2-kernel-bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
