cmake_minimum_required(VERSION 3.16)
project(cpp_implementation)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3)
include_directories(
        ${EIGEN3_INCLUDE_DIR}
)
include_directories(.)

add_executable(cpp_implementation
        kdtree.h
        main.cpp
        octree.h)
