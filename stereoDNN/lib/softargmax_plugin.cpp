// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>
#include <numeric>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// Soft argmax plugin.
// -----------------------------------------------------------------
class SoftargmaxPlugin: public IPlugin
{
public:
    SoftargmaxPlugin(SoftargmaxType sm_type, ILogger& log, std::string name):
        sm_type_(sm_type), log_(log), name_(name)
    {
        // REVIEW alexeyk: FP32 only for now.
        data_type_ = CUDNN_DATA_FLOAT;
        assert(sm_type_ == SoftargmaxType::kMax || sm_type_ == SoftargmaxType::kMin);
    }

    SoftargmaxPlugin(SoftargmaxPlugin&&) = delete;

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        assert(nbInputDims == 1);
        assert(inputs[0].nbDims == 3 || inputs[0].nbDims == 4);
        // If input is 5D tensor in NDCHW format (batch index implicit) then C dim must be equal to 1.
        if (inputs[0].nbDims == 3)
            in_dims_ = inputs[0];
        else
        {
            assert(inputs[0].d[1] == 1);
            in_dims_ = {3, {inputs[0].d[0], inputs[0].d[2], inputs[0].d[3]}};
        }

        out_dims_ = DimsCHW(1, in_dims_.d[1], in_dims_.d[2]);

        return out_dims_;
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
    {
        assert(nbInputs  == 1);
        assert(nbOutputs == 1);
        assert(inputDims[0].nbDims == 3 || inputDims[0].nbDims == 4);
        if (inputDims[0].nbDims == 3)
            assert(DimsUtils::areEqual(inputDims[0],  in_dims_));
        else
            assert(DimsUtils::areEqual(inputDims[0],  Dims{4, {in_dims_.d[0], 1, in_dims_.d[1], in_dims_.d[2]}}));
        assert(DimsUtils::areEqual(outputDims[0], out_dims_));

        last_batch_size_ = maxBatchSize;

        createDescriptors();
        setTensorDescriptors(maxBatchSize);

        log_.log(ILogger::Severity::kINFO, (name_ + ": InDims : " + DimsUtils::toString(in_dims_)).c_str());
        log_.log(ILogger::Severity::kINFO, (name_ + ": OutDims: " + DimsUtils::toString(out_dims_)).c_str());

        assert(isValid());
    }

    int initialize() override
    {
        assert(isValid());

        // Create and copy indices. Not using workspace as amount of memory required
        // for indices is tiny.
        std::vector<float> indices(in_dims_.d[0]);
        std::iota(indices.begin(), indices.end(), 0);
        size_t size_bytes = indices.size() * sizeof(float);
        CHECK(cudaMalloc(&indices_d_, size_bytes));
        CHECK(cudaMemcpy(indices_d_, indices.data(), size_bytes, cudaMemcpyHostToDevice));

        return 0;
    }

    void terminate() override
    {
        assert(isValid());

        if (in_desc_ != nullptr)
            CHECK(cudnnDestroyTensorDescriptor(in_desc_));
        if (out_desc_ != nullptr)
            CHECK(cudnnDestroyTensorDescriptor(out_desc_));
        if (idx_desc_ != nullptr)
            CHECK(cudnnDestroyTensorDescriptor(idx_desc_));
        if (op_desc_ != nullptr)
            CHECK(cudnnDestroyOpTensorDescriptor(op_desc_));
        if (sum_desc_ != nullptr)
            CHECK(cudnnDestroyReduceTensorDescriptor(sum_desc_));
        if (cudnn_ != nullptr)
            CHECK(cudnnDestroy(cudnn_));
        if (indices_d_ != nullptr)
            CHECK(cudaFree(indices_d_));
        in_desc_   = nullptr;
        out_desc_  = nullptr;
        idx_desc_  = nullptr;
        cudnn_     = nullptr;
        indices_d_ = nullptr;
        sum_desc_  = nullptr;

        assert(!isValid());
    }

    size_t getWorkspaceSize(int maxBatchSize) const
    {
        // Need a copy of the input.
        return 2 * (DimsUtils::getTensorSize(in_dims_)) * maxBatchSize * sizeof(float);
    }

    int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        assert(isValid());

        cudnnStatus_t status    = CUDNN_STATUS_SUCCESS;
        cudaError_t   cu_status = cudaSuccess;

        CHECK(status = cudnnSetStream(cudnn_, stream));

        if (batchSize != last_batch_size_)
            setTensorDescriptors(batchSize);

        auto   pdst = static_cast<float*>(workspace);
        // Copy input to workspace.
        size_t in_size       = batchSize * DimsUtils::getTensorSize(in_dims_);
        size_t in_size_bytes = in_size * sizeof(float);
        CHECK(cu_status = cudaMemcpyAsync(pdst, inputs[0], in_size_bytes, cudaMemcpyDeviceToDevice, stream));

        // Scale by -1 in case of softargmin.
        if (sm_type_ == SoftargmaxType::kMin)
        {
            const float kMinusOne = -1.0f;
            CHECK(status = cudnnScaleTensor(cudnn_, in_desc_, pdst, &kMinusOne));
        }

        // Do softmax over D dim.
        CHECK(status = cudnnSoftmaxForward(cudnn_, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, &Consts::kOne, in_desc_, pdst,
                                           &Consts::kZero, in_desc_, pdst));

        // Multiply and sum-reduce over D dim.
        CHECK(status = cudnnOpTensor(cudnn_, op_desc_, &Consts::kOne, in_desc_, pdst, 
                                     &Consts::kOne, idx_desc_, indices_d_,
                                     &Consts::kZero, in_desc_, pdst));
        CHECK(status = cudnnReduceTensor(cudnn_, sum_desc_, nullptr, 0, pdst + in_size, in_size_bytes,
                                         &Consts::kOne,  in_desc_,  pdst,
                                         &Consts::kZero, out_desc_, outputs[0]));

        return (status == CUDNN_STATUS_SUCCESS && cu_status == cudaSuccess) ? 0 : -1;
    }

    size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize(void* buffer) override
    {
    }

private:
    bool isValid() const
    {
        return cudnn_ != nullptr;
    }

    void createDescriptors()
    {
        if (cudnn_ == nullptr)
            CHECK(cudnnCreate(&cudnn_));
        if (in_desc_ == nullptr)
            CHECK(cudnnCreateTensorDescriptor(&in_desc_));
        if (out_desc_ == nullptr)
            CHECK(cudnnCreateTensorDescriptor(&out_desc_));
        if (idx_desc_ == nullptr)
            CHECK(cudnnCreateTensorDescriptor(&idx_desc_));
        if (op_desc_ == nullptr)
            CHECK(cudnnCreateOpTensorDescriptor(&op_desc_));
        if (sum_desc_ == nullptr)
            CHECK(cudnnCreateReduceTensorDescriptor(&sum_desc_));
    }

    void setTensorDescriptors(int batch_size)
    {
        // Set input tensor descriptor as NDHW (add batch dim and remove C dim).
        auto tensor_dims    = Dims{4, {batch_size, in_dims_.d[0], in_dims_.d[1], in_dims_.d[2]}};
        auto tensor_strides = DimsUtils::getStrides(tensor_dims);
        CHECK(cudnnSetTensorNdDescriptor(in_desc_, data_type_, tensor_dims.nbDims, tensor_dims.d, tensor_strides.d));

        // Set output tensor descriptor as NCHW where C == 1.
        tensor_dims    = Dims{4, {batch_size, out_dims_.d[0], out_dims_.d[1], out_dims_.d[2]}};
        tensor_strides = DimsUtils::getStrides(tensor_dims);
        CHECK(cudnnSetTensorNdDescriptor(out_desc_, data_type_, tensor_dims.nbDims, tensor_dims.d, tensor_strides.d));

        // Set index tensor.
        tensor_dims    = Dims{4, {1, in_dims_.d[0], 1, 1}};
        tensor_strides = DimsUtils::getStrides(tensor_dims);
        CHECK(cudnnSetTensorNdDescriptor(idx_desc_, data_type_, tensor_dims.nbDims, tensor_dims.d, tensor_strides.d));
        
        // Set tensor op for index multiplication.
        CHECK(cudnnSetOpTensorDescriptor(op_desc_, CUDNN_OP_TENSOR_MUL, data_type_, CUDNN_PROPAGATE_NAN));

        // Set reduce descriptor.
        CHECK(cudnnSetReduceTensorDescriptor(sum_desc_, CUDNN_REDUCE_TENSOR_ADD, data_type_, CUDNN_PROPAGATE_NAN,
                                             CUDNN_REDUCE_TENSOR_NO_INDICES, CUDNN_32BIT_INDICES));

        last_batch_size_ = batch_size;
    }

private:
    SoftargmaxType sm_type_;

    cudnnDataType_t               data_type_;
    cudnnHandle_t                 cudnn_    = nullptr;
    cudnnTensorDescriptor_t       in_desc_  = nullptr;
    cudnnTensorDescriptor_t       out_desc_ = nullptr;
    cudnnTensorDescriptor_t       idx_desc_ = nullptr;
    cudnnOpTensorDescriptor_t     op_desc_  = nullptr;
    cudnnReduceTensorDescriptor_t sum_desc_ = nullptr;

    Dims    in_dims_;
    DimsCHW out_dims_;
    int     last_batch_size_ = 0;

    float*  indices_d_ = nullptr;

    ILogger&    log_;
    std::string name_;
};

// Factory method.
IPlugin* PluginContainer::createSoftargmaxPlugin(SoftargmaxType sm_type, std::string name)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new SoftargmaxPlugin(sm_type, log_, name));
    return plugins_.back();
}

} }