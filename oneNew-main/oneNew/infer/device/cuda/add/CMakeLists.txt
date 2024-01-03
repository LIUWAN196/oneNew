cmake_minimum_required(VERSION 3.16.1)

project(add_cuda_liu LANGUAGES CUDA CXX)

include(CheckLanguage)
check_language(CUDA)

project(add_cuda_liu)

cuda_add_library(add331 SHARED t2.cu)

