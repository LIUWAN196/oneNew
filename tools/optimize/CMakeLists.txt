cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME optimize)
project(${PROJECTNAME})

if(DEFINED USING_GPU)
    find_package(CUDA REQUIRED)
    include_directories(/usr/local/cuda/include)
    link_directories(/usr/local/cuda/lib64)
endif()

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

add_compile_options(-mfma)

include_directories(${ROOT_DIR_PATH}/host)
include_directories(${ROOT_DIR_PATH}/host/ops)

add_executable(optimize optimize.cpp)
target_link_libraries(optimize dl)
target_link_libraries(optimize Threads::Threads)
target_link_libraries(optimize  ${OpenCV_LIBS})
target_link_libraries(optimize OpenMP::OpenMP_CXX)
target_link_libraries(optimize OpenMP::OpenMP_C)

if(DEFINED USING_GPU)
    target_link_libraries(optimize ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
endif()
