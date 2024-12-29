cmake_minimum_required(VERSION 3.10.1)

project(Gemm_x86)

set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

find_package(OpenMP REQUIRED)
add_compile_options(-mfma)

include_directories(
    ${NN_COMMON})

add_library(Gemm_x86 SHARED gemm.c)

target_link_libraries(Gemm_x86 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(Gemm_x86 PUBLIC OpenMP::OpenMP_C)

