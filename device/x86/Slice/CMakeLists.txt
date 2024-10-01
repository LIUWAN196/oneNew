cmake_minimum_required(VERSION 3.10.1)

project(Slice_x86)
#set(CMAKE_C_FLAGS_RELEASE "-O3")
set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

#set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -g")
#set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3 -g")


include_directories(
        ${NN_COMMON})

add_library(Slice_x86 SHARED slice.c)

