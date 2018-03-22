# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
# Full license terms provided in LICENSE.md file.

# Build with:
# Debug: cmake -DCMAKE_BUILD_TYPE=Debug ..

cmake_minimum_required(VERSION 3.5)

set(default_build_type "Release")

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(TARGET_SUFFIX "_debug")
    message("Using ${TARGET_SUFFIX} suffix.")
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/../bin")

add_subdirectory(lib)
add_subdirectory(sample_app)
add_subdirectory(tests)
