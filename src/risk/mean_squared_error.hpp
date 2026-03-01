#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class MeanSquareRiskImpl final : public RiskMeasureImpl {
    public:
        torch::Tensor forward(const HedgingResult& hedging_result) const override {
            return hedging_result.pnl.square().mean();
        }
};
TORCH_MODULE(MeanSquareRisk);
} // namespace DSO