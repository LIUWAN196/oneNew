cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME model_metrics)
project(${PROJECTNAME})

if(DEFINED USING_GPU)
    find_package(CUDA REQUIRED)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

message(${OpenCV_INCLUDE_DIRS})

include_directories(${ROOT_DIR_PATH}/host)
include_directories(${ROOT_DIR_PATH}/host/ops)

add_executable(model_metrics metrics.cpp)
target_link_libraries(model_metrics dl)
target_link_libraries(model_metrics Threads::Threads)
target_link_libraries(model_metrics  ${OpenCV_LIBS})
target_link_libraries(model_metrics OpenMP::OpenMP_CXX)
target_link_libraries(model_metrics OpenMP::OpenMP_C)

if(DEFINED USING_GPU)
    target_link_libraries(model_metrics ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
endif()


