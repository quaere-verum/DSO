#pragma once
#include "torch/torch.h"
#include "trainers/optimiser.hpp"
#include "core/differentiable_objective.hpp"

namespace DSO {
class LBFGS final : public Optimiser {
    public:
        LBFGS(torch::optim::LBFGS optim)
        : optim_(std::move(optim)) {}

        torch::Tensor step(DSO::GradientEvaluator& eval) {
            auto closure = [&]() -> torch::Tensor {
                return eval.evaluate_and_set_grads();
            };
            return optim_.step(closure);
        }

        std::vector<torch::optim::OptimizerParamGroup> param_groups() const override {
            return optim_.param_groups();
        }
    private:
        torch::optim::LBFGS optim_;
};
} // namespace DSO