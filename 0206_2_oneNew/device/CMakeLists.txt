cmake_minimum_required(VERSION 3.10.1)

project(device)

#set(CMAKE_C_FLAGS_RELEASE "-O3")
set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

add_subdirectory(x86/Add)
add_subdirectory(x86/Conv)
add_subdirectory(x86/GlobalAveragePool)
add_subdirectory(x86/MaxPool)
add_subdirectory(x86/Relu)
add_subdirectory(x86/Softmax)

