cmake_minimum_required(VERSION 3.10.1)

project(Gelu_x86)

find_package(OpenMP REQUIRED)

set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_library(Gelu_x86 SHARED gelu.c)

target_link_libraries(Gelu_x86 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(Gelu_x86 PUBLIC OpenMP::OpenMP_C)

