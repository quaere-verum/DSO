#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include "core/threading.hpp"
#include "features/feature_extractor.hpp"

namespace DSO {

class ControllerImpl : public torch::nn::Module {
public:
    virtual ~ControllerImpl() = default;
    virtual torch::Tensor forward(
        const MarketView& mv,
        const BatchSpec& batch,
        const EvalContext& ctx
    ) = 0;
};
} // namespace DSO
