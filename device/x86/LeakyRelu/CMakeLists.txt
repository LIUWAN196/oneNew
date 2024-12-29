cmake_minimum_required(VERSION 3.10.1)

project(LeakyRelu_x86)

find_package(OpenMP REQUIRED)

set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_library(LeakyRelu_x86 SHARED leaky_relu.c)

target_link_libraries(LeakyRelu_x86 PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(LeakyRelu_x86 PUBLIC OpenMP::OpenMP_C)

