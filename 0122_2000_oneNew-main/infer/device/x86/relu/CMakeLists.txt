cmake_minimum_required(VERSION 3.16.1)

project(relu)

set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_library(Relu SHARED relu.c)

