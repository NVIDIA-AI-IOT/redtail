// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include <ros/ros.h>
#include <ros/service_client.h>
#include <sensor_msgs/Image.h>
#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>

namespace fs = boost::filesystem;

// REVIEW alexeyk: refactor to functor or something prettier?
class CaffeRosTestsCallback
{
public:
    CaffeRosTestsCallback(): dnn_out_(nullptr)
    {
    }

    void dnnCallback(const sensor_msgs::Image::ConstPtr &msg)
    {
        dnn_out_ = msg;
    }

    sensor_msgs::Image::ConstPtr dnn_out_;
};

static boost::shared_ptr<sensor_msgs::Image> readImage(const std::string& filename, const std::string& encoding = "rgb8")
{
    EXPECT_TRUE(encoding == "rgb8" || encoding == "bgr8" || encoding == "bgra8");
    auto img = cv::imread(filename);
    SCOPED_TRACE(filename);
    EXPECT_TRUE(img.cols > 0 && img.rows > 0);
    // Convert image from BGR format used by OpenCV to RGB or BGRA.
    if (encoding == "rgb8")
        cv::cvtColor(img, img, CV_BGR2RGB);
    else if (encoding == "bgra8")
        cv::cvtColor(img, img, CV_BGR2BGRA);
    auto img_msg = boost::make_shared<sensor_msgs::Image>();
    img_msg->encoding = encoding;
    img_msg->width = img.cols;
    img_msg->height = img.rows;
    img_msg->step = img_msg->width * img.channels();
    auto ptr = img.ptr<unsigned char>(0);
    img_msg->data = std::vector<unsigned char>(ptr, ptr + img_msg->step * img_msg->height);
    return img_msg;
}

TEST(CaffeRosTests, TrailNetPredictions)
{
    ros::NodeHandle nh("~");
    std::string test_data_dir;
    nh.param<std::string>("test_data_dir", test_data_dir, "");
    ASSERT_TRUE(fs::exists(test_data_dir));

    CaffeRosTestsCallback t;
    auto dnn_sub = nh.subscribe<sensor_msgs::Image>("/trailnet/dnn/network/output", 1,
                                                    &CaffeRosTestsCallback::dnnCallback, &t);
    const char* camera_topic = "/trailnet/camera/image_raw";
    auto img_pub = nh.advertise<sensor_msgs::Image>(camera_topic, 1);

    // Test images and expected predictions.
    auto images            = std::vector<std::string>{"rot_l.jpg", "rot_c.jpg", "rot_r.jpg", "tran_l.jpg", "tran_r.jpg"};
    float predictions[][6] = {{0.932, 0.060, 0.006, 0.080, 0.848, 0.071},
                              {0.040, 0.958, 0.001, 0.488, 0.375, 0.135},
                              {0.000, 0.027, 0.971, 0.036, 0.407, 0.555},
                              {0.011, 0.988, 0.000, 0.981, 0.008, 0.009},
                              {0.000, 0.855, 0.144, 0.013, 0.031, 0.954}};
    
    // When running using rostest, current directory is $HOME/.ros
    fs::path data_dir{test_data_dir};

    for (size_t i = 0; i < images.size(); i++)
    {
        auto img_msg = readImage((data_dir / images[i]).string());
        // Use image index as a unique timestamp.
        img_msg->header.stamp.sec  = 0;
        img_msg->header.stamp.nsec = (int)i;

        ros::Rate rate(1000);
        // Wait until DNN processes the current messages. There might be multiple messages
        // in the queue so make sure to select the right one based on current index.
        while (ros::ok() && (t.dnn_out_ == nullptr || t.dnn_out_->header.stamp.nsec != i))
        {
            img_pub.publish(img_msg);
            ros::spinOnce();
            rate.sleep();
        }
        
        EXPECT_TRUE(t.dnn_out_ != nullptr);
        auto dnn_out = *t.dnn_out_;
        // The output should be 1x1x6 (HxWxC).
        EXPECT_EQ(dnn_out.width,  1);
        EXPECT_EQ(dnn_out.height, 1);
        // float32, channels == 6.
        EXPECT_EQ(dnn_out.encoding, "32FC6");
        
        auto data  = reinterpret_cast<const float*>(dnn_out.data.data());
        for (int col = 0; col < 6; col++)
        {
            // Must use proper floating point comparison.
            EXPECT_NEAR(data[col], predictions[i][col], 0.001f) << "Values are not equal at (" << i << ", " << col <<")";
        }
    }
}

TEST(CaffeRosTests, TrailNetPredictionsBGR8)
{
    ros::NodeHandle nh("~");
    std::string test_data_dir;
    nh.param<std::string>("test_data_dir", test_data_dir, "");
    ASSERT_TRUE(fs::exists(test_data_dir));

    CaffeRosTestsCallback t;
    auto dnn_sub = nh.subscribe<sensor_msgs::Image>("/trailnet/dnn/network/output", 1,
                                                    &CaffeRosTestsCallback::dnnCallback, &t);
    const char* camera_topic = "/trailnet/camera/image_raw";
    auto img_pub = nh.advertise<sensor_msgs::Image>(camera_topic, 1);

    // Test images and expected predictions.
    auto images            = std::vector<std::string>{"rot_l.jpg", "rot_c.jpg", "rot_r.jpg", "tran_l.jpg", "tran_r.jpg"};
    float predictions[][6] = {{0.932, 0.060, 0.006, 0.080, 0.848, 0.071},
                              {0.040, 0.958, 0.001, 0.488, 0.375, 0.135},
                              {0.000, 0.027, 0.971, 0.036, 0.407, 0.555},
                              {0.011, 0.988, 0.000, 0.981, 0.008, 0.009},
                              {0.000, 0.855, 0.144, 0.013, 0.031, 0.954}};
    
    // When running using rostest, current directory is $HOME/.ros
    fs::path data_dir{test_data_dir};

    for (size_t i = 0; i < images.size(); i++)
    {
        auto img_msg = readImage((data_dir / images[i]).string(), "bgr8");
        // Use image index as a unique timestamp.
        img_msg->header.stamp.sec  = 0;
        img_msg->header.stamp.nsec = (int)i;

        ros::Rate rate(1000);
        // Wait until DNN processes the current messages. There might be multiple messages
        // in the queue so make sure to select the right one based on current index.
        while (ros::ok() && (t.dnn_out_ == nullptr || t.dnn_out_->header.stamp.nsec != i))
        {
            img_pub.publish(img_msg);
            ros::spinOnce();
            rate.sleep();
        }
        
        EXPECT_TRUE(t.dnn_out_ != nullptr);
        auto dnn_out = *t.dnn_out_;
        // The output should be 1x1x6 (HxWxC).
        EXPECT_EQ(dnn_out.width,  1);
        EXPECT_EQ(dnn_out.height, 1);
        // float32, channels == 6.
        EXPECT_EQ(dnn_out.encoding, "32FC6");
        
        auto data  = reinterpret_cast<const float*>(dnn_out.data.data());
        for (int col = 0; col < 6; col++)
        {
            // Must use proper floating point comparison.
            EXPECT_NEAR(data[col], predictions[i][col], 0.001f) << "Values are not equal at (" << i << ", " << col <<")";
        }
    }
}

TEST(CaffeRosTests, TrailNetPredictionsBGRA8)
{
    ros::NodeHandle nh("~");
    std::string test_data_dir;
    nh.param<std::string>("test_data_dir", test_data_dir, "");
    ASSERT_TRUE(fs::exists(test_data_dir));

    CaffeRosTestsCallback t;
    auto dnn_sub = nh.subscribe<sensor_msgs::Image>("/trailnet/dnn/network/output", 1,
                                                    &CaffeRosTestsCallback::dnnCallback, &t);
    const char* camera_topic = "/trailnet/camera/image_raw";
    auto img_pub = nh.advertise<sensor_msgs::Image>(camera_topic, 1);

    // Test images and expected predictions.
    auto images            = std::vector<std::string>{"rot_l.jpg", "rot_c.jpg", "rot_r.jpg", "tran_l.jpg", "tran_r.jpg"};
    float predictions[][6] = {{0.932, 0.060, 0.006, 0.080, 0.848, 0.071},
                              {0.040, 0.958, 0.001, 0.488, 0.375, 0.135},
                              {0.000, 0.027, 0.971, 0.036, 0.407, 0.555},
                              {0.011, 0.988, 0.000, 0.981, 0.008, 0.009},
                              {0.000, 0.855, 0.144, 0.013, 0.031, 0.954}};
    
    // When running using rostest, current directory is $HOME/.ros
    fs::path data_dir{test_data_dir};

    for (size_t i = 0; i < images.size(); i++)
    {
        auto img_msg = readImage((data_dir / images[i]).string(), "bgra8");
        // Use image index as a unique timestamp.
        img_msg->header.stamp.sec  = 0;
        img_msg->header.stamp.nsec = (int)i;

        ros::Rate rate(1000);
        // Wait until DNN processes the current messages. There might be multiple messages
        // in the queue so make sure to select the right one based on current index.
        while (ros::ok() && (t.dnn_out_ == nullptr || t.dnn_out_->header.stamp.nsec != i))
        {
            img_pub.publish(img_msg);
            ros::spinOnce();
            rate.sleep();
        }
        
        EXPECT_TRUE(t.dnn_out_ != nullptr);
        auto dnn_out = *t.dnn_out_;
        // The output should be 1x1x6 (HxWxC).
        EXPECT_EQ(dnn_out.width,  1);
        EXPECT_EQ(dnn_out.height, 1);
        // float32, channels == 6.
        EXPECT_EQ(dnn_out.encoding, "32FC6");
        
        auto data  = reinterpret_cast<const float*>(dnn_out.data.data());
        for (int col = 0; col < 6; col++)
        {
            // Must use proper floating point comparison.
            EXPECT_NEAR(data[col], predictions[i][col], 0.001f) << "Values are not equal at (" << i << ", " << col <<")";
        }
    }
}

TEST(CaffeRosTests, TrailNetPredictionsFP16)
{
    ros::NodeHandle nh("~");
    std::string test_data_dir;
    nh.param<std::string>("test_data_dir", test_data_dir, "");
    ASSERT_TRUE(fs::exists(test_data_dir));

    CaffeRosTestsCallback t;
    auto dnn_sub = nh.subscribe<sensor_msgs::Image>("/trailnet/dnn_fp16/network/output", 1,
                                                    &CaffeRosTestsCallback::dnnCallback, &t);
    const char* camera_topic = "/trailnet/camera_fp16/image_raw";
    auto img_pub = nh.advertise<sensor_msgs::Image>(camera_topic, 1);

    // Test images and expected predictions.
    auto images            = std::vector<std::string>{"rot_l.jpg", "rot_c.jpg", "rot_r.jpg", "tran_l.jpg", "tran_r.jpg"};
    float predictions[][6] = {{0.932, 0.060, 0.006, 0.080, 0.848, 0.071},
                              {0.040, 0.958, 0.001, 0.488, 0.375, 0.135},
                              {0.000, 0.027, 0.971, 0.036, 0.407, 0.555},
                              {0.011, 0.988, 0.000, 0.981, 0.008, 0.009},
                              {0.000, 0.855, 0.144, 0.013, 0.031, 0.954}};
    
    // When running using rostest, current directory is $HOME/.ros
    fs::path data_dir{test_data_dir};

    for (size_t i = 0; i < images.size(); i++)
    {
        auto img_msg = readImage((data_dir / images[i]).string());
        // Use image index as a unique timestamp.
        img_msg->header.stamp.sec  = 0;
        img_msg->header.stamp.nsec = (int)i;

        ros::Rate rate(1000);
        // Wait until DNN processes the current messages. There might be multiple messages
        // in the queue so make sure to select the right one based on current index.
        while (ros::ok() && (t.dnn_out_ == nullptr || t.dnn_out_->header.stamp.nsec != i))
        {
            img_pub.publish(img_msg);
            ros::spinOnce();
            rate.sleep();
        }

        EXPECT_TRUE(t.dnn_out_ != nullptr);
        auto dnn_out = *t.dnn_out_;
        // The output should be 1x1x6 (HxWxC).
        EXPECT_EQ(dnn_out.width,  1);
        EXPECT_EQ(dnn_out.height, 1);
        // float32, channels == 6.
        EXPECT_EQ(dnn_out.encoding, "32FC6");
        
        auto data  = reinterpret_cast<const float*>(dnn_out.data.data());
        for (int col = 0; col < 6; col++)
        {
            // Must use proper floating point comparison.
            // For FP16 the tolerance is higher than for FP32.
            EXPECT_NEAR(data[col], predictions[i][col], 0.02f) << "Values are not equal at (" << i << ", " << col <<")";
        }
    }
}

TEST(CaffeRosTests, TrailNetPredictionsINT8)
{
    ros::NodeHandle nh("~");
    std::string test_data_dir;
    nh.param<std::string>("test_data_dir", test_data_dir, "");
    ASSERT_TRUE(fs::exists(test_data_dir));

    CaffeRosTestsCallback t;
    auto dnn_sub = nh.subscribe<sensor_msgs::Image>("/trailnet/dnn_int8/network/output", 1,
                                                    &CaffeRosTestsCallback::dnnCallback, &t);
    const char* camera_topic = "/trailnet/camera_int8/image_raw";
    auto img_pub = nh.advertise<sensor_msgs::Image>(camera_topic, 1);

    // Test images and expected predictions.
    auto images            = std::vector<std::string>{"rot_l.jpg", "rot_c.jpg", "rot_r.jpg", "tran_l.jpg", "tran_r.jpg"};
    float predictions[][6] = {{0.685, 0.211, 0.104, 0.493, 0.050, 0.466},
                              {0.112, 0.794, 0.093, 0.541, 0.127, 0.332},
                              {0.000, 0.027, 0.971, 0.095, 0.068, 0.836},
                              {0.100, 0.896, 0.000, 0.521, 0.101, 0.377},
                              {0.156, 0.285, 0.558, 0.074, 0.098, 0.827}};
    
    // When running using rostest, current directory is $HOME/.ros
    fs::path data_dir{test_data_dir};

    for (size_t i = 0; i < images.size(); i++)
    {
        auto img_msg = readImage((data_dir / images[i]).string());
        // Use image index as a unique timestamp.
        img_msg->header.stamp.sec  = 0;
        img_msg->header.stamp.nsec = (int)i;

        ros::Rate rate(1000);
        // Wait until DNN processes the current messages. There might be multiple messages
        // in the queue so make sure to select the right one based on current index.
        while (ros::ok() && (t.dnn_out_ == nullptr || t.dnn_out_->header.stamp.nsec != i))
        {
            img_pub.publish(img_msg);
            ros::spinOnce();
            rate.sleep();
        }

        EXPECT_TRUE(t.dnn_out_ != nullptr);
        auto dnn_out = *t.dnn_out_;
        // The output should be 1x1x6 (HxWxC).
        EXPECT_EQ(dnn_out.width,  1);
        EXPECT_EQ(dnn_out.height, 1);
        // float32, channels == 6.
        EXPECT_EQ(dnn_out.encoding, "32FC6");
        
        auto data  = reinterpret_cast<const float*>(dnn_out.data.data());
        for (int col = 0; col < 6; col++)
        {
            // Must use proper floating point comparison.
            // REVIEW alexeyk: the tolerance has to be much higher in INT8.
            EXPECT_NEAR(data[col], predictions[i][col], 0.1f) << "Values are not equal at (" << i << ", " << col <<")";
        }
    }
}

TEST(CaffeRosTests, YoloNetPredictions)
{
    ros::NodeHandle nh("~");
    std::string test_data_dir;
    nh.param<std::string>("test_data_dir", test_data_dir, "");
    ASSERT_TRUE(fs::exists(test_data_dir));

    CaffeRosTestsCallback t;
    auto dnn_sub = nh.subscribe<sensor_msgs::Image>("/yolo/dnn/network/output", 1,
                                                    &CaffeRosTestsCallback::dnnCallback, &t);
    const char* camera_topic = "/yolo/camera/image_raw";
    auto img_pub = nh.advertise<sensor_msgs::Image>(camera_topic, 1);

    // Test images and expected predictions.
    std::string image{"yolo_2_obj.png"};
    float predictions[][6] = {{14, 0.290, 184, 128,  72, 158},
                              {14, 0.660, 529,  84, 105, 239}};

    // When running using rostest, current directory is $HOME/.ros
    fs::path data_dir{test_data_dir};

    auto img_msg = readImage((data_dir / image).string());

    ros::Rate rate(1000);
    // Wait until DNN processes the current messages. There might be multiple messages
    // in the queue so take the first one.
    while (ros::ok() && t.dnn_out_ == nullptr)
    {
        img_pub.publish(img_msg);
        ros::spinOnce();
        rate.sleep();
    }

    EXPECT_TRUE(t.dnn_out_ != nullptr);
    auto dnn_out = *t.dnn_out_;
    // YOLO output is matrix of n x 6.
    EXPECT_EQ(dnn_out.width,  6);
    // There should be 2 objects detected in the test image.
    EXPECT_EQ(dnn_out.height, 2);
    // Single channel, float32.
    EXPECT_EQ(dnn_out.encoding,  "32FC1");

    auto data  = reinterpret_cast<const float*>(dnn_out.data.data());
    for (size_t row = 0; row < dnn_out.height; row++)
    {
        for (size_t col = 0; col < 6; col++)
        {
            // Must use proper floating point comparison.
            EXPECT_NEAR(data[row * 6 + col], predictions[row][col], 0.001f) << "Values are not equal at (" << row << ", " << col <<")";
        }
    }
}

TEST(CaffeRosTests, YoloNetPredictionsFP16)
{
    ros::NodeHandle nh("~");
    std::string test_data_dir;
    nh.param<std::string>("test_data_dir", test_data_dir, "");
    ASSERT_TRUE(fs::exists(test_data_dir));

    CaffeRosTestsCallback t;
    auto dnn_sub = nh.subscribe<sensor_msgs::Image>("/yolo/dnn_fp16/network/output", 1,
                                                    &CaffeRosTestsCallback::dnnCallback, &t);
    const char* camera_topic = "/yolo/camera_fp16/image_raw";
    auto img_pub = nh.advertise<sensor_msgs::Image>(camera_topic, 1);

    // Test images and expected predictions.
    std::string image{"yolo_2_obj.png"};
    float predictions[][6] = {{14, 0.290, 184, 128,  72, 158},
                              {14, 0.660, 529,  84, 105, 239}};

    // When running using rostest, current directory is $HOME/.ros
    fs::path data_dir{test_data_dir};

    auto img_msg = readImage((data_dir / image).string());

    ros::Rate rate(1000);
    // Wait until DNN processes the current messages. There might be multiple messages
    // in the queue so take the first one.
    while (ros::ok() && t.dnn_out_ == nullptr)
    {
        img_pub.publish(img_msg);
        ros::spinOnce();
        rate.sleep();
    }

    EXPECT_TRUE(t.dnn_out_ != nullptr);
    auto dnn_out = *t.dnn_out_;
    // YOLO output is matrix of n x 6.
    EXPECT_EQ(dnn_out.width,  6);
    // There should be 2 objects detected in the test image.
    EXPECT_EQ(dnn_out.height, 2);
    // Single channel, float32.
    EXPECT_EQ(dnn_out.encoding,  "32FC1");

    auto data  = reinterpret_cast<const float*>(dnn_out.data.data());
    for (size_t row = 0; row < dnn_out.height; row++)
    {
        for (size_t col = 0; col < 6; col++)
        {
            // Must use proper floating point comparison.
            // For FP16 the tolerance is higher than for FP32 and will be different
            // for label and probability and coordinates.
            float tolerance = col <= 1 ? 0.02f : 1.0f;
            EXPECT_NEAR(data[row * 6 + col], predictions[row][col], tolerance) << "Values are not equal at (" << row << ", " << col <<")";
        }
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "CaffeRosTests");
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
