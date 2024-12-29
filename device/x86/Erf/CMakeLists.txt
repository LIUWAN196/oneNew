cmake_minimum_required(VERSION 3.10.1)

project(Erf_x86)

find_package(OpenMP REQUIRED)

set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_library(Erf_x86 SHARED erf.c)

target_link_libraries(Erf_x86 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(Erf_x86 PUBLIC OpenMP::OpenMP_C)

