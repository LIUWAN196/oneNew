cmake_minimum_required(VERSION 3.10.1)

project(Einsum_x86)
#set(CMAKE_C_FLAGS_RELEASE "-O3")

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -mavx2 -O3 -g")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -mavx2 -O3 -g")


set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
        ${NN_COMMON})

add_library(Einsum_x86 SHARED einsum.c)

