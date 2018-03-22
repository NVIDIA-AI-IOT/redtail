// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// ELU activation function plugin.
// -----------------------------------------------------------------
class EluPlugin: public IPlugin
{
public:
    EluPlugin(DataType data_type, ILogger& log, std::string name):
        data_type_(trtToCudnnDataType(data_type)), log_(log), name_(name)
    {
        // REVIEW alexeyk: TRT currently does not support FP16 data tensors.
        assert(data_type == DataType::kFLOAT);
    }

    EluPlugin(EluPlugin&&) = delete;

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        in_dims_  = inputs[0];
        // No restrictions on input dims.

        return in_dims_;
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
    {
        assert(nbInputs == 1);
        assert(nbOutputs == 1);
        // Sanity check.
        assert(DimsUtils::areEqual(in_dims_, inputDims[0]));
        assert(DimsUtils::areEqual(in_dims_, outputDims[0]));

        cudnnStatus_t status;

        CHECK(status = cudnnCreate(&cudnn_));
        CHECK(status = cudnnCreateActivationDescriptor(&act_));
        CHECK(status = cudnnSetActivationDescriptor(act_, CUDNN_ACTIVATION_ELU, CUDNN_PROPAGATE_NAN, 1.0));
        CHECK(status = cudnnCreateTensorDescriptor(&t_desc_));

        setTensorDescriptor(maxBatchSize);

        log_.log(ILogger::Severity::kINFO, (name_ + ": Dims: " + DimsUtils::toString(in_dims_)).c_str());
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
        assert(isValid());

        if (act_ != nullptr)
            CHECK(cudnnDestroyActivationDescriptor(act_));
        if (t_desc_ != nullptr)
            CHECK(cudnnDestroyTensorDescriptor(t_desc_));
        if (cudnn_ != nullptr)
            CHECK(cudnnDestroy(cudnn_));
        act_    = nullptr;
        t_desc_ = nullptr;
        cudnn_  = nullptr;

        assert(!isValid());
    }

    size_t getWorkspaceSize(int maxBatchSize) const
    {
        return 0;
    }

    int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        assert(isValid());

        cudnnStatus_t status;

        CHECK(status = cudnnSetStream(cudnn_, stream));
        if (batchSize != tensor_dims_.d[0])
            updateTensorDescriptor(batchSize);
        CHECK(status = cudnnActivationForward(cudnn_, act_, &Consts::kOne, t_desc_, inputs[0],  &Consts::kZero, t_desc_, outputs[0]));

        return status == CUDNN_STATUS_SUCCESS ? 0 : -1;
    }

    size_t getSerializationSize() override
    {
        assert(isValid());
        return 0;
    }

    void serialize(void* buffer) override
    {
        assert(isValid());
    }

private:
    void setTensorDescriptor(int batch_size)
    {
        assert(isValid());

        tensor_dims_.nbDims = in_dims_.nbDims + 1;
        tensor_dims_.d[0]   = batch_size;
        std::copy(in_dims_.d, in_dims_.d + in_dims_.nbDims, tensor_dims_.d + 1);

        tensor_strides_ = DimsUtils::getStrides(tensor_dims_);

        CHECK(cudnnSetTensorNdDescriptor(t_desc_, data_type_, tensor_dims_.nbDims, tensor_dims_.d, tensor_strides_.d));
    }

    // Updates tensor descriptor according to batch_size.
    void updateTensorDescriptor(int batch_size)
    {
        // No other parameters require update.
        tensor_dims_.d[0] = batch_size;
        CHECK(cudnnSetTensorNdDescriptor(t_desc_, data_type_, tensor_dims_.nbDims, tensor_dims_.d, tensor_strides_.d));
    }

private:
    bool isValid() const
    {
        return cudnn_ != nullptr;
    }

private:
    cudnnDataType_t             data_type_;
    cudnnHandle_t               cudnn_  = nullptr;
    cudnnActivationDescriptor_t act_    = nullptr;
    cudnnTensorDescriptor_t     t_desc_ = nullptr;

    Dims in_dims_;
    Dims tensor_dims_;
    Dims tensor_strides_;

    ILogger&    log_;
    std::string name_;
};

// Factory method.
IPlugin* PluginContainer::createEluPlugin(DataType data_type, std::string name)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new EluPlugin(data_type, log_, name));
    return plugins_.back();
}

} }