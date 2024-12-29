cmake_minimum_required(VERSION 3.10.1)

project(ConvTranspose_x86)

find_package(OpenMP REQUIRED)
add_compile_options(-mfma)

set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
        ${NN_COMMON})

add_library(ConvTranspose_x86 SHARED conv_transpose.c)

target_link_libraries(ConvTranspose_x86 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(ConvTranspose_x86 PUBLIC OpenMP::OpenMP_C)

