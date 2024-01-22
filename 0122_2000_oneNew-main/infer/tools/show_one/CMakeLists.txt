cmake_minimum_required(VERSION 3.16.1)
set(PROJECTNAME one_new)
project(${PROJECTNAME})

set(CMAKE_BUILD_TYPE DEBUG)
add_executable(show_one show_one_info.cpp)

