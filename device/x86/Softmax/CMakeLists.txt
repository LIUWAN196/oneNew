cmake_minimum_required(VERSION 3.10.1)

project(Softmax_x86)

set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
        ${NN_COMMON})

add_library(Softmax_x86 SHARED softmax.c)

