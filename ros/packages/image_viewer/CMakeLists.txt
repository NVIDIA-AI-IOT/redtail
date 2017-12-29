cmake_minimum_required(VERSION 2.8)
project(image_viewer)

find_package(catkin REQUIRED COMPONENTS camera_calibration_parsers cv_bridge dynamic_reconfigure image_transport message_filters message_generation nodelet rosconsole roscpp std_srvs stereo_msgs)
generate_dynamic_reconfigure_options(cfg/ImageView.cfg)

catkin_package(CATKIN_DEPENDS dynamic_reconfigure)
find_package(Boost REQUIRED COMPONENTS signals thread)
find_package(OpenCV 3 REQUIRED)

include_directories(${Boost_INCLUDE_DIRS}
                    ${catkin_INCLUDE_DIRS}
                    ${OpenCV_INCLUDE_DIRS}
)

# Extra tools

add_executable(image_saver src/nodes/image_saver.cpp)
target_link_libraries(image_saver ${catkin_LIBRARIES}
                                  ${OpenCV_LIBRARIES}
)



# Deal with the GUI's
if(ANDROID)
  return()
endif()

find_package(GTK2)
add_definitions(-DHAVE_GTK)
include_directories(${GTK2_INCLUDE_DIRS})

# Nodelet library
add_library(image_viewer src/nodelets/image_nodelet.cpp src/nodelets/disparity_nodelet.cpp src/nodelets/window_thread.cpp)
target_link_libraries(image_viewer ${catkin_LIBRARIES}
                                 ${GTK_LIBRARIES}
                                 ${GTK2_LIBRARIES}
                                 ${OpenCV_LIBRARIES}
                                 ${Boost_LIBRARIES}
)
install(TARGETS image_viewer
        DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
)

# Image viewers
add_executable(image_viewer_exe src/nodes/image_viewer.cpp)
add_dependencies(image_viewer_exe ${PROJECT_NAME}_gencfg)
SET_TARGET_PROPERTIES(image_viewer_exe PROPERTIES OUTPUT_NAME image_viewer)
target_link_libraries(image_viewer_exe ${catkin_LIBRARIES}
                                     ${OpenCV_LIBRARIES}
                                     ${Boost_LIBRARIES}
)



install(TARGETS image_viewer_exe
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(FILES nodelet_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

