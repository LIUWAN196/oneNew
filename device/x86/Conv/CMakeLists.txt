cmake_minimum_required(VERSION 3.10.1)

project(Conv_x86)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

set(NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
        ${NN_COMMON})

add_library(Conv_x86 SHARED conv.c)

target_link_libraries(Conv_x86 Threads::Threads)
target_link_libraries(Conv_x86 OpenMP::OpenMP_CXX)
target_link_libraries(Conv_x86 OpenMP::OpenMP_C)

