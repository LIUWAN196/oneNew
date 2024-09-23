cmake_minimum_required(VERSION 3.10.1)

project(device)

set(CMAKE_CXX_FLAGS_DEBUG "-std=c++11 -mavx2 -O3 -g")
set(CMAKE_C_FLAGS_DEBUG "-std=c11 -mavx2 -O3 -g")

set (NN_COMMON "/home/e0006809/Desktop/oneNew-main/infer/common")

include_directories(
    ${NN_COMMON})

# step 1: compiler x86 kernel
# 获取该文件夹下的子文件夹名称
 file(GLOB kernel_path ${CMAKE_CURRENT_SOURCE_DIR}/x86/*) # 获取目标路径下的内容，深度为 1
 set(KERNEL_LIST "")
 foreach (kernel ${kernel_path})
     file(RELATIVE_PATH kernel_name ${CMAKE_CURRENT_SOURCE_DIR}/x86 ${kernel}) # 通过相对路径计算获取名称
     # 获取到子文件的名称，然后使用 IS_DIRECTORY 判断是否为子文件夹而不是文件
     if (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/x86/${kernel_name})
        list(APPEND KERNEL_LIST ${kernel_name})
     endif ()
 endforeach ()

message(STATUS "KERNEL_LIST: ${KERNEL_LIST}")

foreach(kernel ${KERNEL_LIST})    
    add_subdirectory(x86/${kernel})
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