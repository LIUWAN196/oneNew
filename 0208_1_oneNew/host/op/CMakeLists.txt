cmake_minimum_required(VERSION 3.10.1)
set(PROJECTNAME net_test)
project(${PROJECTNAME})

#find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

#message(${OpenCV_INCLUDE_DIRS})

add_executable(net_1 test_net.cpp)
target_link_libraries(net_1 dl)
target_link_libraries(net_1 Threads::Threads)
#target_link_libraries( net_1  ${OpenCV_LIBS})




