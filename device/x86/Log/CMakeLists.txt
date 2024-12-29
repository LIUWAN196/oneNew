cmake_minimum_required(VERSION 3.10.1)

project(Log_x86)

find_package(OpenMP REQUIRED)

set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_library(Log_x86 SHARED log.c)

target_link_libraries(Log_x86 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(Log_x86 PUBLIC OpenMP::OpenMP_C)

