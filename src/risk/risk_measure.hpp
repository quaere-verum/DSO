#pragma once
#include "hedging/hedging.hpp"

namespace DSO {

class RiskMeasure {
    public:
        virtual ~RiskMeasure() = default;
        virtual torch::Tensor evaluate(const HedgingResult& hedging_result) const = 0;
};

} // namespace DSO