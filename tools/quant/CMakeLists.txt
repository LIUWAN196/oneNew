cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME quant)
project(${PROJECTNAME})

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${ROOT_DIR_PATH}/host)
include_directories(${ROOT_DIR_PATH}/host/ops)

add_executable(quant quant_conv_one.cpp)
target_link_libraries(quant dl)
target_link_libraries(quant Threads::Threads)
target_link_libraries(quant  ${OpenCV_LIBS})
target_link_libraries(quant OpenMP::OpenMP_CXX)
target_link_libraries(quant OpenMP::OpenMP_C)

