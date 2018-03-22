cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME nvstereo_tests)
project(${PROJECT_NAME})

find_package(CUDA 9.0 REQUIRED)
find_package(OpenCV 3.3.1 REQUIRED)

set(GTEST_ROOT "/usr/src/gtest")
find_package(GTest REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${GTEST_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/lib)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

file(GLOB ${PROJECT_NAME}_sources ./*.cpp)
set(PROJECT_SOURCES ${${PROJECT_NAME}_sources})

set(TARGET_NAME ${PROJECT_NAME}${TARGET_SUFFIX})

add_executable(${TARGET_NAME} ${PROJECT_SOURCES})

target_link_libraries(${TARGET_NAME}
    nvstereo_inference${TARGET_SUFFIX}
    ${CUDA_LIBRARIES}
    cudnn
    nvinfer
    ${GTEST_LIBRARIES}
    pthread
    opencv_core)
