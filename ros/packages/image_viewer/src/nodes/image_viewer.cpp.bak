/*********************************************************************
* Software License Agreement (BSD License)
*
*  Copyright (c) 2008, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/
#include <image_view/ImageViewConfig.h>

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <dynamic_reconfigure/server.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>

#include <boost/format.hpp>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>

int g_count;
cv::Mat g_last_image;
boost::format g_filename_format;
boost::mutex g_image_mutex;
std::string g_window_name;
bool g_gui;
ros::Publisher g_pub;
bool g_do_dynamic_scaling;
int g_colormap;
double g_min_image_value;
double g_max_image_value;

void reconfigureCb(image_view::ImageViewConfig &config, uint32_t level)
{
  boost::mutex::scoped_lock lock(g_image_mutex);
  g_do_dynamic_scaling = config.do_dynamic_scaling;
  g_colormap = config.colormap;
  g_min_image_value = config.min_image_value;
  g_max_image_value = config.max_image_value;
}

void imageCb(const sensor_msgs::ImageConstPtr& msg)
{
  boost::mutex::scoped_lock lock(g_image_mutex);

  // Convert to OpenCV native BGR color
  cv_bridge::CvImageConstPtr cv_ptr;
  try {
    cv_bridge::CvtColorForDisplayOptions options;
    options.do_dynamic_scaling = g_do_dynamic_scaling;
    options.colormap = g_colormap;
    // Set min/max value for scaling to visualize depth/float image.
    if (g_min_image_value == g_max_image_value) {
      // Not specified by rosparam, then set default value.
      // Because of current sensor limitation, we use 10m as default of max range of depth
      // with consistency to the configuration in rqt_image_view.
      options.min_image_value = 0;
      if (msg->encoding == "32FC1") {
        options.max_image_value = 10;  // 10 [m]
      } else if (msg->encoding == "16UC1") {
        options.max_image_value = 10 * 1000;  // 10 * 1000 [mm]
      }
    } else {
      options.min_image_value = g_min_image_value;
      options.max_image_value = g_max_image_value;
    }
    cv_ptr = cv_bridge::cvtColorForDisplay(cv_bridge::toCvShare(msg), "", options);
    g_last_image = cv_ptr->image;
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR_THROTTLE(30, "Unable to convert '%s' image for display: '%s'",
                       msg->encoding.c_str(), e.what());
  }
  if (g_gui && !g_last_image.empty()) {
    const cv::Mat &image = g_last_image;
    cv::imshow(g_window_name, image);
    cv::waitKey(3);
  }
  if (g_pub.getNumSubscribers() > 0) {
    g_pub.publish(cv_ptr);
  }
}

static void mouseCb(int event, int x, int y, int flags, void* param)
{
  if (event == cv::EVENT_LBUTTONDOWN) {
    ROS_WARN_ONCE("Left-clicking no longer saves images. Right-click instead.");
    return;
  } else if (event != cv::EVENT_RBUTTONDOWN) {
    return;
  }

  boost::mutex::scoped_lock lock(g_image_mutex);

  const cv::Mat &image = g_last_image;

  if (image.empty()) {
    ROS_WARN("Couldn't save image, no data!");
    return;
  }

  std::string filename = (g_filename_format % g_count).str();
  if (cv::imwrite(filename, image)) {
    ROS_INFO("Saved image %s", filename.c_str());
    g_count++;
  } else {
    boost::filesystem::path full_path = boost::filesystem::complete(filename);
    ROS_ERROR_STREAM("Failed to save image. Have permission to write there?: " << full_path);
  }
}


int main(int argc, char **argv)
{
  ros::init(argc, argv, "image_view", ros::init_options::AnonymousName);
  if (ros::names::remap("image") == "image") {
    ROS_WARN("Topic 'image' has not been remapped! Typical command-line usage:\n"
             "\t$ rosrun image_view image_view image:=<image topic> [transport]");
  }

  ros::NodeHandle nh;
  ros::NodeHandle local_nh("~");

  // Default window name is the resolved topic name
  std::string topic = nh.resolveName("image");
  local_nh.param("window_name", g_window_name, topic);
  local_nh.param("gui", g_gui, true);  // gui/no_gui mode

  if (g_gui) {
    std::string format_string;
    local_nh.param("filename_format", format_string, std::string("frame%04i.jpg"));
    g_filename_format.parse(format_string);

    // Handle window size
    bool autosize;
    local_nh.param("autosize", autosize, false);
    cv::namedWindow(g_window_name, autosize ? (CV_WINDOW_AUTOSIZE | CV_WINDOW_KEEPRATIO | CV_GUI_EXPANDED) : 0);
    cv::setMouseCallback(g_window_name, &mouseCb);

    if(autosize == false)
    {
      if(local_nh.hasParam("width") && local_nh.hasParam("height"))
      {
        int width;
        local_nh.getParam("width", width);
        int height;
        local_nh.getParam("height", height);
        cv::resizeWindow(g_window_name, width, height);
      }
    }

    // Start the OpenCV window thread so we don't have to waitKey() somewhere
    cv::startWindowThread();
  }

  // Handle transport
  // priority:
  //    1. command line argument
  //    2. rosparam '~image_transport'
  std::string transport;
  local_nh.param("image_transport", transport, std::string("raw"));
  ros::V_string myargv;
  ros::removeROSArgs(argc, argv, myargv);
  for (size_t i = 1; i < myargv.size(); ++i) {
    if (myargv[i][0] != '-') {
      transport = myargv[i];
      break;
    }
  }
  ROS_INFO_STREAM("Using transport \"" << transport << "\"");
  image_transport::ImageTransport it(nh);
  image_transport::TransportHints hints(transport, ros::TransportHints(), local_nh);
  image_transport::Subscriber sub = it.subscribe(topic, 1, imageCb, hints);
  g_pub = local_nh.advertise<sensor_msgs::Image>("output", 1);

  dynamic_reconfigure::Server<image_view::ImageViewConfig> srv;
  dynamic_reconfigure::Server<image_view::ImageViewConfig>::CallbackType f =
    boost::bind(&reconfigureCb, _1, _2);
  srv.setCallback(f);

  ros::spin();

  if (g_gui) {
    cv::destroyWindow(g_window_name);
  }
  return 0;
}
