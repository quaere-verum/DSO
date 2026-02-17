#pragma once
#include "torch/torch.h"
#include "trainers/optimiser.hpp"
#include "core/differentiable_objective.hpp"

namespace DSO {
class LBFGS final : public Optimiser {
    public:
        LBFGS(torch::optim::LBFGS optim)
        : optim_(std::move(optim)) {}

        torch::Tensor step(DSO::DifferentiableObjective& objective) {
            auto closure = [&]() -> torch::Tensor {
                optim_.zero_grad();
                auto loss = objective.forward();
                loss.backward();
                return loss;
            };
            return optim_.step(closure);
        }
    private:
        torch::optim::LBFGS optim_;
};
} // namespace DSO