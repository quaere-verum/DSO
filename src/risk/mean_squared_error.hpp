#include "risk/risk_measure.hpp"
#include "hedging/hedging.hpp"
#include <torch/torch.h>

namespace DSO {
class MeanSquareRisk final : public RiskMeasure {
    public:
        torch::Tensor evaluate(const HedgingResult& hedging_result) const override {
            return hedging_result.pnl.square().mean();
        }
    };
} // namespace DSO