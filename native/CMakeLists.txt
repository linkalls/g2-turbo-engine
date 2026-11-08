cmake_minimum_required(VERSION 3.22)
project(g2_turbo LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ANDROID)
    # One pinned llama.cpp CPU backend for actual GGUF inference.
    # Keep the standalone KleidiAI comparison below; avoid duplicate
    # kai_* symbols by disabling GGML_CPU_KLEIDIAI in this baseline.
    set(LLAMA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp")
    if(NOT EXISTS "${LLAMA_SOURCE_DIR}/include/llama.h")
        message(FATAL_ERROR "Pinned llama.cpp missing. Run scripts/fetch-llama.sh")
    endif()
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "" FORCE)
    set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod" CACHE STRING "" FORCE)
    set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    add_subdirectory("${LLAMA_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/llama" EXCLUDE_FROM_ALL)

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
        src/kernel_bench.cpp src/cpu_pool.cpp src/vulkan_bench.cpp src/kleidi_bench.cpp
        src/gguf_inference.cpp src/gguf_sweep.cpp)
    target_compile_options(g2turbo PRIVATE -O3 -ffp-contract=fast -march=armv8.2-a+dotprod)
    target_include_directories(g2turbo PRIVATE "${LLAMA_SOURCE_DIR}/include")
    target_link_libraries(g2turbo PRIVATE llama g2kleidi log android vulkan)
else()
    add_executable(g2-kernel-bench src/kernel_bench.cpp src/cpu_pool.cpp)
    target_compile_options(g2-kernel-bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
