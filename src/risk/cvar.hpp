#pragma once
#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class CVaRRiskImpl final : public RiskMeasureImpl {
    public:
        explicit CVaRRiskImpl(double alpha, bool trainable_z = false)
            : alpha_(alpha)
            , trainable_z_(trainable_z) {
                if (trainable_z_) z_ = register_parameter("z", torch::zeros({1}, torch::TensorOptions().dtype(torch::kFloat32)));
            }

        torch::Tensor forward(const HedgingResult& hedging_result) override {
            const auto& pnl = hedging_result.pnl;
            torch::Tensor loss = -pnl;  // treat losses positive
            torch::Tensor z;
            if (trainable_z_) {
                z = z_;
            } else {
                z = torch::quantile(loss.detach(), alpha_);
            }
            torch::Tensor tail = torch::relu(loss - z);
            return z + tail.mean() / (1.0 - alpha_);
        }

    private:
        double alpha_;
        bool trainable_z_;
        torch::Tensor z_;
};
TORCH_MODULE(CVaRRisk);
} // namespace DSO