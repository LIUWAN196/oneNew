cmake_minimum_required(VERSION 3.10.1)

project(MatMul_x86)

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -mavx2 -O3 -g")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -mavx2 -O3 -g")

find_package(OpenMP REQUIRED)
add_compile_options(-mfma)

#set(CMAKE_C_FLAGS_RELEASE "-O3")
set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_library(MatMul_x86 SHARED mat_mul.c)

target_link_libraries(MatMul_x86 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(MatMul_x86 PUBLIC OpenMP::OpenMP_C)

