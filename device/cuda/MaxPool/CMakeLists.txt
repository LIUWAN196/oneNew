cmake_minimum_required(VERSION 3.10.1)

project(MaxPool_cu)

set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
        ${NN_COMMON})

cuda_add_library(MaxPool_cu SHARED maxpool.cu)
