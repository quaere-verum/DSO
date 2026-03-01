#pragma once
#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class CVaRRiskImpl final : public RiskMeasureImpl {
    public:
        explicit CVaRRiskImpl(double alpha)
            : alpha_(alpha) {
                z_ = register_parameter("z", torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32)));
            }

        torch::Tensor forward(const HedgingResult& hedging_result) const override {
            const auto& pnl = hedging_result.pnl;
            torch::Tensor loss = -pnl;  // treat losses positive

            torch::Tensor tail = torch::relu(loss - z_);
            return z_ + tail.mean() / (1.0 - alpha_);
        }

    private:
        double alpha_;
        torch::Tensor z_;
};
TORCH_MODULE(CVaRRisk);
} // namespace DSO