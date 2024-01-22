cmake_minimum_required(VERSION 3.16.1)
set(PROJECTNAME auto_kernel_host)
project(${PROJECTNAME})

add_executable(cpp_host cpp_test.c)
target_link_libraries(cpp_host dl)