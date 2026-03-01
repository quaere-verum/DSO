#pragma once
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {

class RiskMeasureImpl : public torch::nn::Module {
    public:
        virtual ~RiskMeasureImpl() = default;
        virtual torch::Tensor forward(const HedgingResult& hedging_result) const = 0;
};

} // namespace DSO