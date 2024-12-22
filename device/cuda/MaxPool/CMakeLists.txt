cmake_minimum_required(VERSION 3.10.1)

project(MaxPool_cu)

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -mavx2")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -mavx2")

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
        ${NN_COMMON})


cuda_add_library(MaxPool_cu SHARED maxpool.cu)
#cuda_add_library(MaxPool_x86 SHARED MaxPool.cu)

#target_link_libraries(MaxPool_cu PUBLIC OpenMP::OpenMP_CXX)
#target_link_libraries(MaxPool_cu PUBLIC OpenMP::OpenMP_C)

