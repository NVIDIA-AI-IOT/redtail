cmake_minimum_required(VERSION 3.1)
set (CMAKE_CXX_STANDARD 14)

project( widecam_mono_calibration )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( widecam_mono_calibration main.cpp )
target_link_libraries( widecam_mono_calibration ${OpenCV_LIBS} )

