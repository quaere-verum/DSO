#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class EntropicRiskImpl final : public RiskMeasureImpl {
    public:
        explicit EntropicRiskImpl(double gamma)
            : gamma_(gamma) {}

        torch::Tensor forward(const HedgingResult& hedging_result) const override {
            const auto& pnl = hedging_result.pnl;
            torch::Tensor scaled = gamma_ * pnl;

            // numerical stabilisation
            torch::Tensor max_val = std::get<0>(scaled.max(0));
            torch::Tensor stabilized = torch::exp(scaled - max_val);
            torch::Tensor mean = stabilized.mean();

            return (max_val + mean.log()) / gamma_;
        }

    private:
        double gamma_;
};
TORCH_MODULE(EntropicRisk);
} // namespace DSO