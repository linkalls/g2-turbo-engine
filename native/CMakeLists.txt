cmake_minimum_required(VERSION 3.22)
project(g2_turbo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(ANDROID)
    add_library(g2turbo SHARED src/kernel_bench.cpp src/vulkan_bench.cpp)
    target_compile_options(g2turbo PRIVATE -O3 -ffp-contract=fast -march=armv8.2-a+dotprod)
    target_link_libraries(g2turbo PRIVATE log android vulkan)
else()
    add_executable(g2-kernel-bench src/kernel_bench.cpp)
    target_compile_options(g2-kernel-bench PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()
