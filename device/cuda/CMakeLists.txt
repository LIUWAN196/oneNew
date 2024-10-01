cmake_minimum_required(VERSION 3.10.1)

project(device/cuda)

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

