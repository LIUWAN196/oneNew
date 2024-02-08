cmake_minimum_required(VERSION 3.10.1)

project(Gemm_x86)

#set(CMAKE_C_FLAGS_RELEASE "-O3")
set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_library(Gemm_x86 SHARED gemm.c)

