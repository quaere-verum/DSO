#pragma once
#include "torch/torch.h"
#include "trainers/optimiser.hpp"
#include "core/differentiable_objective.hpp"

namespace DSO {
class Adam final : public Optimiser {
    public:
        Adam(torch::optim::Adam optim)
        : optim_(std::move(optim)) {}

        torch::Tensor step(DSO::GradientEvaluator& eval) {
            optim_.zero_grad();
            torch::Tensor loss = eval.evaluate_and_set_grads();
            optim_.step();
            return loss;
        }
        void set_lr(double lr) {
            for (auto& p_group : optim_.param_groups()) {
                p_group.options().set_lr(lr);
            }
        }
    private:
        torch::optim::Adam optim_;
};
} // namespace DSO