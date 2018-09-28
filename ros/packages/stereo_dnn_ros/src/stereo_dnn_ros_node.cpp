// Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include <unordered_map>

#include <boost/algorithm/string.hpp>
#include <boost/format.hpp>

#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include <ros/ros.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>

#include "redtail_tensorrt_plugins.h"
#include "networks.h"

#define UNUSED(x) ((void)(x))

#define CHECK(status) do {   \
    int res = (int)(status); \
    assert(res == 0);        \
    UNUSED(res);             \
} while(false)

using namespace nvinfer1;
using namespace redtail::tensorrt;

using ConstStr = const std::string;

namespace stereo_dnn_ros
{

static sensor_msgs::Image::ConstPtr s_cur_img_l = nullptr;
static sensor_msgs::Image::ConstPtr s_cur_img_r = nullptr;

cv::Mat preprocessImage(cv::Mat img, int dst_img_w, int dst_img_h, ConstStr& encoding)
{
    // Handle encodings.
    if (encoding == "bgr8")
        cv::cvtColor(img, img, CV_BGR2RGB);
    else if (encoding == "bgra8")
        cv::cvtColor(img, img, CV_BGRA2RGB);
    //ROS_INFO("Dims: (%zu, %zu) -> (%zu, %zu)", w, h, (size_t)dst_img_w, (size_t)dst_img_h);
    // Convert to floating point type.
    img.convertTo(img, CV_32F);
    // Resize (anisotropically) to input layer size.
    cv::resize(img, img, cv::Size(dst_img_w, dst_img_h), 0, 0, cv::INTER_AREA);
    // Scale.
    img /= 255.0;
    // Transpose to get CHW format.
    return img.reshape(1, dst_img_w * dst_img_h).t();
}

sensor_msgs::Image::ConstPtr computeOutputs(IExecutionContext *context, size_t h, size_t w,
                                            int idx_l, int idx_r, int idx_out, void** buffers)
{
    if (s_cur_img_l == nullptr || s_cur_img_r == nullptr)
        return nullptr;

    size_t c = 3;
    sensor_msgs::ImageConstPtr imgs[] {s_cur_img_l,  s_cur_img_r};
    void* bufs[] {buffers[idx_l], buffers[idx_r]};
    for (int i = 0; i < 2; i++)
    {
        auto img      = *(imgs[i]);
        auto img_h    = cv::Mat((int)img.height, (int)img.width, img.encoding == "bgra8" ? CV_8UC4 : CV_8UC3, (void*)img.data.data());
        auto final_h_ = preprocessImage(img_h, w, h, img.encoding);
        CHECK(cudaMemcpy(bufs[i], final_h_.data, c * h * w * sizeof(float), cudaMemcpyHostToDevice));
    }

    auto err = context->execute(1, buffers);
    assert(err);
    auto output = cv::Mat((int)h, (int)w, CV_32FC1);
    CHECK(cudaMemcpy(output.data, buffers[idx_out], h * w * sizeof(float), cudaMemcpyDeviceToHost));
    output *= w;

    auto out_msg = boost::make_shared<sensor_msgs::Image>();
    // Set stamp and frame id to the same value as source image so we can synchronize with other nodes if needed.
    auto img_l = *s_cur_img_l;
    out_msg->header.stamp.sec  = img_l.header.stamp.sec;
    out_msg->header.stamp.nsec = img_l.header.stamp.nsec;
    out_msg->header.frame_id   = img_l.header.frame_id;
    out_msg->encoding = "32FC1";
    out_msg->width    = w;
    out_msg->height   = h;
    out_msg->step     = out_msg->width * sizeof(float);
    size_t count      = out_msg->step * out_msg->height;
    auto ptr          = reinterpret_cast<const unsigned char*>(output.data);
    out_msg->data     = std::vector<unsigned char>(ptr, ptr + count);
    // ROS_INFO("computeOutputs: %u, %u, %s", out_msg->width, out_msg->height, out_msg->encoding.c_str());

    // Set to null to mark as completed.
    s_cur_img_l = nullptr;
    s_cur_img_r = nullptr;

    return out_msg;
}

//void imageCallback(const sensor_msgs::Image::ConstPtr& msg_l, const sensor_msgs::Image::ConstPtr& msg_r)
void imageCallback(const sensor_msgs::ImageConstPtr& msg_l, const sensor_msgs::ImageConstPtr& msg_r)
{
    auto img_l = *msg_l;
    auto img_r = *msg_r;
    // ROS_INFO("imageCallback: %u, %u, %s", img_l.width, img_l.height, img_l.encoding.c_str());
    // ROS_INFO("imageCallback: %u, %u, %s", img_r.width, img_r.height, img_r.encoding.c_str());
    if (img_l.encoding != "rgb8" && img_l.encoding != "bgr8" && img_l.encoding != "bgra8")
    {
        ROS_FATAL("Image encoding %s is not yet supported. Supported encodings: rgb8, bgr8, bgra8", img_l.encoding.c_str());
        ros::shutdown();
    }
    if (img_r.encoding != "rgb8" && img_r.encoding != "bgr8" && img_r.encoding != "bgra8")
    {
        ROS_FATAL("Image encoding %s is not yet supported. Supported encodings: rgb8, bgr8, bgra8", img_r.encoding.c_str());
        ros::shutdown();
    }

    s_cur_img_l = msg_l;
    s_cur_img_r = msg_r;
}

void parseModelType(const std::string& src, int& h, int& w)
{
    if (boost::iequals(src, "nvsmall"))
    {
        h = 321;
        w = 1025;
    }
    else if (boost::iequals(src, "nvtiny"))
    {
        h = 161;
        w = 513;
    }
    else if (boost::iequals(src, "resnet18"))
    {
        h = 321;
        w = 1025;
    }
    else if (boost::iequals(src, "resnet18_2D"))
    {
        h = 257;
        w = 513;
    }
    else
    {
        ROS_FATAL("Not supported model type: %s. Supported types: nvsmall, nvtiny, resnet18, resnet18_2D", src.c_str());
        ros::shutdown();
    }
}

DataType parseDataType(const std::string& src)
{
    if (boost::iequals(src, "FP32"))
        return DataType::kFLOAT;
    if (boost::iequals(src, "FP16"))
        return DataType::kHALF;
    else
    {
        ROS_FATAL("Invalid data type: %s. Supported data types: FP32, FP16.", src.c_str());
        ros::shutdown();
        // Will not get here (well, should not).
        return (DataType)-1;
    }
}

std::unordered_map<std::string, Weights> readWeights(const std::string& filename, DataType data_type)
{
    assert(data_type == DataType::kFLOAT || data_type == DataType::kHALF);

    std::unordered_map<std::string, Weights> weights;
    std::ifstream weights_file(filename, std::ios::binary);
    assert(weights_file.is_open());
    while (weights_file.peek() != std::ifstream::traits_type::eof())
    {
        std::string name;
        uint32_t    count;
        Weights     w {data_type, nullptr, 0};
        std::getline(weights_file, name, '\0');
        weights_file.read(reinterpret_cast<char*>(&count), sizeof(uint32_t));
        w.count = count;
        size_t el_size_bytes = data_type == DataType::kFLOAT ? 4 : 2;
        auto p = new uint8_t[count * el_size_bytes];
        weights_file.read(reinterpret_cast<char*>(p), count * el_size_bytes);
        w.values = p;
        assert(weights.find(name) == weights.cend());
        weights[name] = w;
    }
    return weights;
}

} // stereo_dnn_ros

namespace sd = stereo_dnn_ros;
namespace mf = message_filters;

class Logger : public ILogger
{
public:
    void log(ILogger::Severity severity, const char* msg) override
    {
        // Skip info (verbose) messages.
        if (severity == Severity::kINFO)
            return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "TRT INTERNAL_ERROR: "; break;
            case Severity::kERROR:          std::cerr << "TRT ERROR: "; break;
            case Severity::kWARNING:        std::cerr << "TRT WARNING: "; break;
            case Severity::kINFO:           std::cerr << "TRT INFO: "; break;
            default:                        std::cerr << "TRT UNKNOWN: "; break;
        }
        std::cerr << msg << std::endl;
    }
};

static Logger gLogger;

int main(int argc, char **argv)
{
    ros::init(argc, argv, "stereo_dnn_ros");

    ROS_INFO("Starting Stereo DNN ROS node...");
    ros::NodeHandle nh("~");

    std::string camera_topic_l;
    std::string camera_topic_r;
    std::string model_type;
    std::string model_path;
    std::string data_type_s;
    int         camera_queue_size;
    int         dnn_queue_size;
    float       max_rate_hz;
    bool        debug_mode;

    nh.param<std::string>("camera_topic_left",  camera_topic_l, "/zed/left/image_rect_color");
    nh.param<std::string>("camera_topic_right", camera_topic_r, "/zed/right/image_rect_color");
    nh.param<std::string>("model_type", model_type, "resnet18_2D");
    nh.param<std::string>("model_path", model_path, "");
    nh.param<std::string>("data_type",  data_type_s, "fp16");

    nh.param("camera_queue_size", camera_queue_size, 2);
    nh.param("dnn_queue_size",    dnn_queue_size,    2);
    nh.param("max_rate_hz",       max_rate_hz, 30.0f);
    nh.param("debug_mode",        debug_mode,  false);

    int c = 3;
    int h = 0;
    int w = 0;

    sd::parseModelType(model_type, h, w);

    ROS_INFO("Camera L: %s", camera_topic_l.c_str());
    ROS_INFO("Camera R: %s", camera_topic_r.c_str());
    ROS_INFO("Model T : %s", model_type.c_str());
    ROS_INFO("Model   : %s", model_path.c_str());
    ROS_INFO("DType   : %s", data_type_s.c_str());
    ROS_INFO("Cam Q   : %d", camera_queue_size);
    ROS_INFO("DNN Q   : %d", dnn_queue_size);
    ROS_INFO("Rate    : %.1f", max_rate_hz);
    ROS_INFO("Debug   : %s", debug_mode ? "yes" : "no");

    auto data_type = sd::parseDataType(data_type_s);

    // TensorRT pre-built plan file.
    auto trt_plan_file = model_path + ".plan";
    std::ifstream trt_plan(trt_plan_file, std::ios::binary);

    // Note: the plugin_container object lifetime must be at least the same as the engine.
    auto plugin_container = IPluginContainer::create(gLogger);
    ICudaEngine* engine   = nullptr;

    // Check if we can load pre-built model from TRT plan file.
    // Currently only ResNet18_2D supports serialization.
    if (model_type == "resnet18_2D" && trt_plan.good())
    {
        ROS_INFO("Loading TensorRT plan from %s...", trt_plan_file.c_str());
        // StereoDnnPluginFactory object is stateless as it adds plugins to corresponding container.
        StereoDnnPluginFactory factory(*plugin_container);
        IRuntime* runtime = createInferRuntime(gLogger);
        // Load the plan.
        std::stringstream model;
        model << trt_plan.rdbuf();
        model.seekg(0, model.beg);
        const auto& model_final = model.str();
        // Deserialize model.
        engine = runtime->deserializeCudaEngine(model_final.c_str(), model_final.size(), &factory);
        runtime->destroy();
    }
    else
    {
        ROS_INFO("Loading TensorRT weights from %s...", model_path.c_str());
        // Read weights.
        auto weights = sd::readWeights(model_path, data_type);

        // Create builder and network.
        IBuilder* builder = createInferBuilder(gLogger);

        // For now only ResNet18_2D has proper support for FP16.
        INetworkDefinition* network = nullptr;
        if (model_type == "nvsmall")
            network = createNVSmall1025x321Network(    *builder, *plugin_container, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
        else if (model_type == "nvtiny")
            network = createNVTiny513x161Network(      *builder, *plugin_container, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
        else if (model_type == "resnet18")
            network = createResNet18_1025x321Network(  *builder, *plugin_container, DimsCHW { c, h, w }, weights, DataType::kFLOAT, gLogger);
        else if (model_type == "resnet18_2D")
            network = createResNet18_2D_513x257Network(*builder, *plugin_container, DimsCHW { c, h, w }, weights, data_type, gLogger);
        else
            ROS_ASSERT(false);

        builder->setMaxBatchSize(1);
        size_t workspace_bytes = 1024 * 1024 * 1024;
        builder->setMaxWorkspaceSize(workspace_bytes);

        builder->setHalf2Mode(data_type == DataType::kHALF);
        // Build the network.
        engine = builder->buildCudaEngine(*network);
        // Cleanup.
        network->destroy();
        builder->destroy();

        if (model_type == "resnet18_2D")
        {
            ROS_INFO("Saving TensorRT plan to %s...", trt_plan_file.c_str());
            IHostMemory *model_stream = engine->serialize();
            std::ofstream trt_plan_out(trt_plan_file, std::ios::binary);
            trt_plan_out.write((const char*)model_stream->data(), model_stream->size());
        }
    }

    assert(engine->getNbBindings() == 3);
    void* buffers[3];
    int in_idx_left = engine->getBindingIndex("left");
    assert(in_idx_left == 0);
    int in_idx_right = engine->getBindingIndex("right");
    assert(in_idx_right == 1);
    int out_idx = engine->getBindingIndex("disp");
    assert(out_idx == 2);

    IExecutionContext *context = engine->createExecutionContext();

    // if (debug_mode_)
    //     net_.showProfile(true);

    mf::Subscriber<sensor_msgs::Image> image_sub_l(nh, camera_topic_l, camera_queue_size);
    mf::Subscriber<sensor_msgs::Image> image_sub_r(nh, camera_topic_r, camera_queue_size);

    using MySyncPolicy = mf::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image>;
    mf::Synchronizer<MySyncPolicy> sync(MySyncPolicy(camera_queue_size), image_sub_l, image_sub_r);
    //mf::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image> sync(image_sub_l, image_sub_r, 10);
    sync.registerCallback(boost::bind(&sd::imageCallback, _1, _2));

    auto output_pub = nh.advertise<sensor_msgs::Image>("network/output", dnn_queue_size);

    size_t img_size = c * h * w;
    CHECK(cudaMalloc(&buffers[in_idx_left],  img_size * sizeof(float)));
    CHECK(cudaMalloc(&buffers[in_idx_right], img_size * sizeof(float)));
    CHECK(cudaMalloc(&buffers[out_idx],      img_size * sizeof(float)));

    ros::Rate rate(max_rate_hz);
    ros::spinOnce();
    while (ros::ok())
    {
        auto out_msg = sd::computeOutputs(context, h, w, in_idx_left, in_idx_right, out_idx, buffers);
        if (out_msg != nullptr)
            output_pub.publish(out_msg);
        ros::spinOnce();
        rate.sleep();
    }
    return 0;
}
