#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class MeanVarianceRisk final : public RiskMeasure {
    public:
        explicit MeanVarianceRisk(double lambda)
            : lambda_(lambda) {}

        torch::Tensor evaluate(const HedgingResult& hedging_result) const override {
            const auto& pnl = hedging_result.pnl;
            auto mean = pnl.mean();
            auto var  = (pnl - mean).square().mean();
            return mean + lambda_ * var;
        }

    private:
        double lambda_;
};
} // namespace DSO