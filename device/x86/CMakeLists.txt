cmake_minimum_required(VERSION 3.10.1)

project(device/x86)


# 根据 CMAKE_BUILD_TYPE 设置不同的编译选项
if(CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    # Debug 版本使用 O1 优化并支持调试 -g
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -std=c++11 -mavx2 -O1 -g")
    set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -std=c11 -mavx2 -O1 -g")
elseif(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    # Release 版本使用 O3 优化，不支持 -g
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -std=c++11 -mavx2 -O3")
    set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -std=c11 -mavx2 -O3")
#    add_compile_options(-mfma)
endif ()


#set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -mavx2 -O3 -g")
#set(CMAKE_C_FLAGS_DEBUG "-std=c11 -mavx2 -O3 -g")
#find_package(OpenMP REQUIRED)
add_compile_options(-mfma)

#set(x86_lib_dir ${CMAKE_CURRENT_BINARY_DIR})
#add_definitions(-DOP_X86_LIB_DIR=x86_lib_dir)
#message(STATUS "aaa OP_X86_LIB_DIR: ${CMAKE_CURRENT_BINARY_DIR}")
##message(STATUS "bbb OP_X86_LIB_DIR: ${BUILD_DIR_NAME}")

# 使用 GLOB_RECURSE 获取当前目录下的所有内容（深度为 1）
file(GLOB children RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)

# 初始化空列表
set(KERNEL_LIST "")

# 遍历找到的所有项
foreach(child ${children})
    # 构建完整路径
    set(child_path ${CMAKE_CURRENT_SOURCE_DIR}/${child})

    # 检查是否为目录
    if(EXISTS ${child_path} AND IS_DIRECTORY ${child_path})
        # 将目录名添加到列表中
        list(APPEND KERNEL_LIST ${child})
    endif()
endforeach()

foreach(kernel ${KERNEL_LIST})    
    add_subdirectory(${kernel})
endforeach()


## step 1: compiler cuda kernel
## 获取该文件夹下的子文件夹名称
#file(GLOB kernel_path ${CMAKE_CURRENT_SOURCE_DIR}/cuda/*) # 获取目标路径下的内容，深度为 1
#set(KERNEL_LIST "")
#foreach (kernel ${kernel_path})
#    file(RELATIVE_PATH kernel_name ${CMAKE_CURRENT_SOURCE_DIR}/x86 ${kernel}) # 通过相对路径计算获取名称
#    # 获取到子文件的名称，然后使用 IS_DIRECTORY 判断是否为子文件夹而不是文件
#    if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/x86/${kernel_name})
#        list(APPEND KERNEL_LIST ${kernel_name})
#    endif ()
#endforeach ()
#
#message(STATUS "KERNEL_LIST: ${KERNEL_LIST}")
#
#foreach(kernel ${KERNEL_LIST})
#    add_subdirectory(cuda/${kernel})
#endforeach()