# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME nvstereo_inference)
project(${PROJECT_NAME})

find_package(CUDA 10.0 REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# Set CUDA NVCC flags:
# Enable C++ 14
list(APPEND CUDA_NVCC_FLAGS -std=c++14)
list(APPEND CUDA_NVCC_FLAGS -lineinfo -Xcompiler -Wall -Xcompiler -Wextra)

# Generate both PTX and SASS for Jetson TX2 and use base Maxwell (TX1) PTX for everything else.
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_53,code=compute_53)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_61)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_62,code=sm_62)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_70,code=sm_70)
list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_72,code=sm_72)

file(GLOB ${PROJECT_NAME}_sources ./*.cpp ./*.cu)
set(PROJECT_SOURCES ${${PROJECT_NAME}_sources})

set(TARGET_NAME ${PROJECT_NAME}${TARGET_SUFFIX})
cuda_add_library(${TARGET_NAME} ${PROJECT_SOURCES} STATIC)
