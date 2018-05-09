// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Full license terms provided in LICENSE.md file.

#include "internal_utils.h"
#include <cassert>

namespace redtail { namespace tensorrt
{

using namespace nvinfer1;

// -----------------------------------------------------------------
// Cost volume plugin.
// -----------------------------------------------------------------
class CostVolumePlugin: public IPlugin
{
public:
    CostVolumePlugin(int max_disparity, ILogger& log, std::string name):
        max_disparity_(max_disparity), log_(log), name_(name)
    {
        assert(max_disparity_ > 0);
    }

    CostVolumePlugin(CostVolumePlugin&&) = delete;

    int getNbOutputs() const override
    {
        return 1;
    }

    Dims getOutputDimensions(int index, const Dims* inputs, int nbInputDims) override
    {
        // Expecting 2 3D inputs, left and right, which are outputs of 2D convolutions
        // and are of the same shape.
        assert(nbInputDims == 2);
        for (int i = 0; i < nbInputDims; i++)
        {
            assert(inputs[i].nbDims == 3);
            assert(inputs[i].type[0] == DimensionType::kCHANNEL);
            assert(inputs[i].type[1] == DimensionType::kSPATIAL);
            assert(inputs[i].type[2] == DimensionType::kSPATIAL);
        }
        assert(DimsUtils::areEqual(inputs[0], inputs[1]));

        in_dims_  = inputs[0];
        out_dims_ = DimsNCHW(max_disparity_, 2 * in_dims_.d[0], in_dims_.d[1], in_dims_.d[2]);
        return out_dims_;
    }

    void configure(const Dims* inputDims, int nbInputs, const Dims* outputDims, int nbOutputs, int maxBatchSize) override
    {
        assert(nbInputs  == 2);
        assert(nbOutputs == 1);
        assert(DimsUtils::areEqual(inputDims[0],  in_dims_));
        assert(DimsUtils::areEqual(inputDims[1],  in_dims_));
        assert(DimsUtils::areEqual(outputDims[0], out_dims_));
        assert(maxBatchSize == 1);

        log_.log(ILogger::Severity::kINFO, (name_ + ": InDims(x2): " + DimsUtils::toString(in_dims_)).c_str());
        log_.log(ILogger::Severity::kINFO, (name_ + ": OutDims   : " + DimsUtils::toString(out_dims_)).c_str());
    }

    int initialize() override
    {
        return 0;
    }

    void terminate() override
    {
    }

    size_t getWorkspaceSize(int maxBatchSize) const
    {
        assert(maxBatchSize == 1);
        return 0;
    }

    int enqueue(int batchSize, const void*const * inputs, void** outputs, void* workspace, cudaStream_t stream) override
    {
        assert(batchSize == 1);

        auto pleft     = static_cast<const float*>(inputs[0]);
        auto pright    = static_cast<const float*>(inputs[1]);
        auto pcost_vol = static_cast<float*>(outputs[0]);

        cudaError_t status;
        CHECK(status = CudaKernels::computeCostVolume(pleft, pright, in_dims_, pcost_vol, out_dims_, stream));

        return status;
    }

    size_t getSerializationSize() override
    {
        return 0;
    }

    void serialize(void* buffer) override
    {
    }

private:
    int      max_disparity_;
    Dims     in_dims_;
    DimsNCHW out_dims_;
    
    ILogger&    log_;
    std::string name_;
};

// Factory method.
IPlugin* PluginContainer::createCostVolumePlugin(int max_disparity, std::string name)
{
    std::lock_guard<std::mutex> lock(lock_);
    plugins_.push_back(new CostVolumePlugin(max_disparity, log_, name));
    return plugins_.back();
}

} }