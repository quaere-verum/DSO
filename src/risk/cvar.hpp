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

            int64_t B = loss.size(0);
            int64_t k = static_cast<int64_t>(std::ceil(alpha_ * B));

            auto sorted = std::get<0>(loss.sort());
            auto tail = sorted.slice(0, k, B);

            return tail.mean();
        }

    private:
        double alpha_;
};
} // namespace DSO