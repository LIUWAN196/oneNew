cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME optimize)
project(${PROJECTNAME})

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${ROOT_DIR_PATH}/host)
include_directories(${ROOT_DIR_PATH}/host/ops)

add_executable(optimize optimize.cpp)
target_link_libraries(optimize dl)
target_link_libraries(optimize Threads::Threads)
target_link_libraries(optimize  ${OpenCV_LIBS})
target_link_libraries(optimize OpenMP::OpenMP_CXX)
target_link_libraries(optimize OpenMP::OpenMP_C)

