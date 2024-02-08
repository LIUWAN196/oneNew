cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME example)
project(${PROJECTNAME})

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(${ROOT_DIR_PATH}/host)
include_directories(${ROOT_DIR_PATH}/host/ops)

add_executable(model_infer model_infer.cpp)
target_link_libraries(model_infer dl)
target_link_libraries(model_infer Threads::Threads)
target_link_libraries(model_infer  ${OpenCV_LIBS})
target_link_libraries(model_infer OpenMP::OpenMP_CXX)
target_link_libraries(model_infer OpenMP::OpenMP_C)



