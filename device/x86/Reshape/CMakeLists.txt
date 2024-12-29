cmake_minimum_required(VERSION 3.10.1)

project(Reshape_x86)

set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_library(Reshape_x86 SHARED reshape.c)

