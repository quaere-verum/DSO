#pragma once
#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class CVaRRisk final : public RiskMeasure {
    public:
        explicit CVaRRisk(double alpha)
            : alpha_(alpha) {}

        torch::Tensor evaluate(const HedgingResult& hedging_result) const override {
            const auto& pnl = hedging_result.pnl;
            torch::Tensor loss = -pnl;  // treat losses positive

            torch::Tensor z = torch::quantile(loss.detach(), alpha_);
            torch::Tensor tail = torch::relu(loss - z);
            return z + tail.mean() / (1.0 - alpha_);
        }

    private:
        double alpha_;
};
} // namespace DSO