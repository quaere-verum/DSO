#pragma once
#include <torch/torch.h>
#include <memory>
#include "core/stochastic_program.hpp"
#include "products/product.hpp"
#include "control/controller.hpp"
#include "core/threading.hpp"
#include "core/time_grid.hpp"
#include "hedging/hedging.hpp"
#include "risk/risk_measure.hpp"

namespace DSO {
class MCHedgeObjective final : public StochasticProgram {
    public:
        MCHedgeObjective(
            size_t n_paths,
            const Product& product,
            const ControllerImpl& controller,
            const HedgingEngine& hedging_engine,
            const RiskMeasure& risk_measure,
            const FeatureExtractorImpl& feature_extractor
        )
        : n_paths_(n_paths)
        , product_(product)
        , controller_(controller)
        , hedging_engine_(hedging_engine)
        , risk_measure_(risk_measure)
        , feature_extractor_(feature_extractor) {}

        torch::Tensor loss(
            const SimulationResult& simulated,
            const BatchSpec& batch,
            const EvalContext& ctx
        ) override {
            HedgingResult result = hedging_engine_.run(simulated, product_, controller_, feature_extractor_);
            torch::Tensor risk = risk_measure_.evaluate(result);
            return risk;
        }

        void resample_paths(size_t n_paths) override {
            n_paths_ = n_paths;
            ++epoch_;
            epoch_rng_offset_ = static_cast<uint64_t>(epoch_) * (1ULL << 32);
        }

        size_t n_paths() const { return n_paths_; }
        uint64_t epoch_rng_offset() const { return epoch_rng_offset_; }

    private:
        size_t n_paths_;
        const Product& product_;
        const ControllerImpl& controller_;
        const HedgingEngine& hedging_engine_;
        const RiskMeasure& risk_measure_;
        const FeatureExtractorImpl& feature_extractor_;

        size_t epoch_ = 0;
        uint64_t epoch_rng_offset_ = 0;
};

} // namespace DSO
