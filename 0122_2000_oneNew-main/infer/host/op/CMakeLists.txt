cmake_minimum_required(VERSION 3.16.1)
set(PROJECTNAME net_test)
project(${PROJECTNAME})



add_executable(net_1 test_net.cpp)
target_link_libraries(net_1 dl)