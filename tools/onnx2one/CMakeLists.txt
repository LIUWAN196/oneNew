find_package(protobuf CONFIG)

set(CMAKE_BUILD_TYPE DEBUG)

find_package(OpenMP REQUIRED)

if(protobuf_FOUND)
    set(PROTOBUF_FOUND ${protobuf_FOUND})
    set(PROTOBUF_VERSION ${protobuf_VERSION})
else()
    # fallback to system
    find_package(Protobuf)
    set(PROTOBUF_FOUND ${Protobuf_FOUND})
    set(PROTOBUF_VERSION ${Protobuf_VERSION})
    if(TARGET protobuf::protoc)
        set_target_properties(protobuf::protoc PROPERTIES IMPORTED_LOCATION_RELEASE "${PROTOBUF_PROTOC_EXECUTABLE}")
    endif()
endif()

include_directories(${ROOT_DIR_PATH}/host)
include_directories(${ROOT_DIR_PATH}/host/ops)

if(PROTOBUF_FOUND)
    if(DEFINED PROTOBUF_VERSION AND PROTOBUF_VERSION VERSION_GREATER_EQUAL 3.22)
        set(CMAKE_CXX_STANDARD 17)
    endif()

    if(Protobuf_FOUND OR protobuf_MODULE_COMPATIBLE)
        protobuf_generate_cpp(ONNX_PROTO_SRCS ONNX_PROTO_HDRS onnx.proto)
        add_executable(onnx2one onnx2one.cpp ${ONNX_PROTO_SRCS} ${ONNX_PROTO_HDRS})
        target_include_directories(onnx2one PRIVATE ${PROTOBUF_INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
        target_link_libraries(onnx2one PRIVATE ${PROTOBUF_LIBRARIES})

    else()
        add_executable(onnx2one onnx2one.cpp onnx.proto)
        target_include_directories(onnx2one PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
        protobuf_generate(TARGET onnx2one)
        target_link_libraries(onnx2one PRIVATE protobuf::libprotobuf)
    endif()

else()
    message(WARNING "Protobuf not found, onnx model convert tool won't be built")
endif()

target_link_libraries(onnx2one PRIVATE dl)
target_link_libraries(onnx2one PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(onnx2one PRIVATE OpenMP::OpenMP_C)

