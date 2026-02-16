#pragma once
#include "torch/torch.h"
#include "trainers/optimiser.hpp"
#include "core/differentiable_objective.hpp"

namespace DSO {
class Adam final : public Optimiser {
    public:
        Adam(torch::optim::Adam optim)
        : optim_(std::move(optim)) {}

        void step(DSO::DifferentiableObjective& objective) {
            optim_.zero_grad();
            auto loss = objective.forward();
            loss.backward();
            optim_.step();
        }
    private:
        torch::optim::Adam optim_;
};
} // namespace DSO