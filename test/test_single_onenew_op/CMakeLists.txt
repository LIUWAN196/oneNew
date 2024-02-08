cmake_minimum_required(VERSION 3.16.1)
set(PROJECTNAME relu_op_test)
project(${PROJECTNAME})



add_executable(relu test_relu.cpp)
target_link_libraries(relu dl)