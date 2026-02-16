#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>

#include "core/stochastic_program.hpp"
#include "models/stochastic_model.hpp"
#include "products/product.hpp"
#include "trainers/adam_wrapper.hpp"

namespace DSO {

class MCPriceObjective final : public StochasticProgram {
public:
    MCPriceObjective(
        double target_price,
        size_t n_paths,
        StochasticModel& model,
        const Product& product
    )
    : n_paths_(n_paths)
    , model_(model)
    , product_(product) {
        TORCH_CHECK(model.factors() == product.factors(), "model factors must  be same as product factors")
        auto opt = torch::TensorOptions().dtype(torch::kFloat32);
        target_price_ = torch::tensor({target_price}, opt);
        param_names_ = model_.parameter_names();
    }

    torch::Tensor forward() override {
        auto paths = model_.simulate_paths(n_paths_, product_, rng_stream_offset_);
        if (!payoffs_.defined() || payoffs_.numel() != paths.size(0)) {
            payoffs_ = torch::empty({paths.size(0)}, paths.options().dtype(torch::kFloat32));
        }
        product_.compute_payoff(paths, payoffs_);

        // TODO: Add discounting
        torch::Tensor price = payoffs_.mean();
        auto diff = price - target_price_;
        return diff * diff; 
    }

    void resample_paths(size_t n_paths) override {
        static constexpr uint64_t stride = 1ULL << 32;
        rng_stream_offset_ = (epoch_++) * stride;
        n_paths_ = n_paths;
    }

    std::vector<torch::Tensor> parameters() override {
        return model_.parameters();
    }

    const std::vector<std::string>& parameter_names() const override {
        return param_names_;
    }

private:
    torch::Tensor target_price_;
    torch::Tensor payoffs_;

    size_t n_paths_;
    StochasticModel& model_;
    const Product& product_;
    std::vector<std::string> param_names_;

    size_t rng_stream_offset_ = 0;
    size_t epoch_ = 0;
};

} // namespace DSO
