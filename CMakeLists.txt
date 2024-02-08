cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME oneNew)
project(${PROJECTNAME})
set(CMAKE_BUILD_TYPE DEBUG)

set(ROOT_DIR_PATH ${CMAKE_CURRENT_LIST_DIR})

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O3 -g")
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O3 -g")

add_subdirectory(device)

add_subdirectory(example)

add_subdirectory(tools/quant)
add_subdirectory(tools/metrics)
add_subdirectory(tools/onnx2one)
add_subdirectory(tools/optimize)

