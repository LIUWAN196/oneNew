cmake_minimum_required(VERSION 3.10.1)

project(GlobalAveragePool_x86)

set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
        ${NN_COMMON})

add_library(GlobalAveragePool_x86 SHARED global_avgpool.c)

