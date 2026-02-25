#pragma once
#include <torch/torch.h>
#include "core/stochastic_program.hpp"
#include "models/stochastic_model.hpp"
#include "products/product.hpp"
#include "core/threading.hpp"

namespace DSO {
class MCCalibrationObjective final : public StochasticProgram {
    public:
        MCCalibrationObjective(
            double target_price,
            size_t n_paths,
            const Product& product
        )
        : n_paths_(n_paths)
        , product_(product) {
            auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            target_price_ = torch::tensor({static_cast<float>(target_price)}, opt);
        }

        torch::Tensor loss(
            const SimulationResult& simulated,
            const BatchSpec& /*batch*/,
            const EvalContext& /*ctx*/
        ) override {
            torch::Tensor payoffs = product_.compute_payoff(simulated);
            return payoffs.sub(target_price_).mean().square();
        }

        void resample_paths(size_t n_paths) override {
            n_paths_ = n_paths;
            ++epoch_;
            epoch_rng_offset_ = static_cast<uint64_t>(epoch_) * (1ULL << 32);
        }

        size_t n_paths() const { return n_paths_; }
        uint64_t epoch_rng_offset() const { return epoch_rng_offset_; }

    private:
        torch::Tensor target_price_;
        size_t n_paths_;

        const Product& product_;

        size_t epoch_ = 0;
        uint64_t epoch_rng_offset_ = 0;
};
} // namespace DSO