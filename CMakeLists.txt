cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME oneNew)
project(${PROJECTNAME})

set(ROOT_DIR_PATH ${CMAKE_CURRENT_LIST_DIR})

# 默认为 Debug 版本
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build, options are: Debug Release." FORCE)
endif()

# 转换 CMAKE_BUILD_TYPE 为大写，以便统一处理
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_CMAKE_BUILD_TYPE)

# 根据 CMAKE_BUILD_TYPE 设置不同的编译选项
if(UPPER_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    message("current CMAKE_BUILD_TYPE: ${UPPER_CMAKE_BUILD_TYPE}")
    # Debug 版本使用 O1 优化并支持调试 -g
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -O0 -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -std=c11 -O0 -g")
elseif(UPPER_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    message("current CMAKE_BUILD_TYPE: ${UPPER_CMAKE_BUILD_TYPE}")
    # Release 版本使用 O3 优化，不支持 -g
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -O3")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -std=c11 -O3")
else()
    message(FATAL_ERROR "Only support DEBUG and RELEASE, Unsupported CMAKE_BUILD_TYPE: ${UPPER_CMAKE_BUILD_TYPE}")
endif()

# 支持单独设置 LOG_LEV
if(DEFINED LOG_LEV)
    if(LOG_LEV STREQUAL "1")
        add_definitions(-DLOG_LEV=1)
    elseif(LOG_LEV STREQUAL "2")
        add_definitions(-DLOG_LEV=2)
    elseif(LOG_LEV STREQUAL "3")
        add_definitions(-DLOG_LEV=3)
    elseif(LOG_LEV STREQUAL "0")
        # 取消定义 LOG_LEV 宏
        unset(LOG_LEV)
    else()
        message(FATAL_ERROR "LOG_LEV only support 0 1 2 3, Unsupported LOG_LEV: ${LOG_LEV}")
    endif()
else()
    # 如果没有单独设置 LOG_LEV，则使用默认值
    if(UPPER_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        add_definitions(-DLOG_LEV=2)
    elseif(UPPER_CMAKE_BUILD_TYPE STREQUAL "RELEASE")
        add_definitions(-DLOG_LEV=1)
    endif()
endif()

message("fffff current LOG_LEV: ${LOG_LEV}")

set(x86_lib_dir ${CMAKE_CURRENT_BINARY_DIR}/device/x86)
message(STATUS "aaa OP_X86_LIB_DIR: ${x86_lib_dir}")
add_definitions(-DOP_X86_LIB_DIR="${x86_lib_dir}")

add_subdirectory(device/x86)

add_subdirectory(example)

add_subdirectory(tools/metrics)
add_subdirectory(tools/onnx2one)
add_subdirectory(tools/optimize)

