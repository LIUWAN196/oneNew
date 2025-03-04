cmake_minimum_required(VERSION 3.16.1)

project(cuda_demo LANGUAGES CUDA CXX)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O2")

include(CheckLanguage)
check_language(CUDA)

add_executable(cuda_demo main_test.cu ops/reduce.cu ops/transpose.cu ops/gemm.cu)

