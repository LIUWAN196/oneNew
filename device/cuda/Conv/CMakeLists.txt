cmake_minimum_required(VERSION 3.10.1)

project(Conv_cu)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2 -O3")
set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mavx2 -O3")

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -mavx2 -O3")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -mavx2 -O3")

find_package(CUDA REQUIRED)
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

find_package(OpenMP REQUIRED)
add_compile_options(-mfma)

#set(CMAKE_C_FLAGS_RELEASE "-O3")
set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
        ${NN_COMMON})

cuda_add_library(Conv_cu SHARED conv.cu)
#cuda_add_library(Conv_x86 SHARED conv.cu)

#target_link_libraries(Conv_cu PUBLIC OpenMP::OpenMP_CXX)
#target_link_libraries(Conv_cu PUBLIC OpenMP::OpenMP_C)

