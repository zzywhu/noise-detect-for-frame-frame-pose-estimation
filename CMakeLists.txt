
### CMakeLists.txt


cmake_minimum_required(VERSION 3.10)
project(OutlierDetection)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable(OutlierDetection main.cpp)

