cmake_minimum_required(VERSION 2.6)

project(LP_CORNER_DETECT)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(CUDA REQUIRED)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include ${PROJECT_SOURCE_DIR}/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86)")
    message(STATUS "current arch is x86")
    include_directories(/root/TensorRT-7.0.0.11/include)
    link_directories(/root/TensorRT-7.0.0.11/lib)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED -Wno-error=deprecated-declarations -Wno-deprecated-declarations -DOCRTEST")

add_executable(lp_corner_detect ${PROJECT_SOURCE_DIR}/src/corner_detect.cpp)
target_link_libraries(lp_corner_detect nvinfer cudart ${OpenCV_LIBS})
