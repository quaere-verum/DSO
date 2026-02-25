#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class EntropicRisk final : public RiskMeasure {
    public:
        explicit EntropicRisk(double gamma)
            : gamma_(gamma) {}

        torch::Tensor evaluate(const HedgingResult& hedging_result) const override {
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
} // namespace DSO