#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class MeanVarianceRiskImpl final : public RiskMeasureImpl {
    public:
        explicit MeanVarianceRiskImpl(double lambda)
            : lambda_(lambda) {}

        torch::Tensor forward(const HedgingResult& hedging_result) const override {
            const auto& pnl = hedging_result.pnl;
            auto mean = pnl.mean();
            auto var  = (pnl - mean).square().mean();
            return mean + lambda_ * var;
        }

    private:
        double lambda_;
};
TORCH_MODULE(MeanVarianceRisk);
} // namespace DSO