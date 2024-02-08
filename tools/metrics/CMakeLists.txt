cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME net_metrics)
project(${PROJECTNAME})

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

message(${OpenCV_INCLUDE_DIRS})

include_directories(${ROOT_DIR_PATH}/host)
include_directories(${ROOT_DIR_PATH}/host/ops)

add_executable(net_metrics net_metrics.cpp)
target_link_libraries(net_metrics dl)
target_link_libraries(net_metrics Threads::Threads)
target_link_libraries(net_metrics  ${OpenCV_LIBS})
target_link_libraries(net_metrics OpenMP::OpenMP_CXX)
target_link_libraries(net_metrics OpenMP::OpenMP_C)



