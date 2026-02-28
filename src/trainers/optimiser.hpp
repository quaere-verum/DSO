#pragma once
#include "core/differentiable_objective.hpp"
#include <torch/torch.h>

namespace DSO {
class GradientEvaluator {
    public:
        virtual ~GradientEvaluator() = default;
        virtual torch::Tensor evaluate_and_set_grads() = 0;
};
class Optimiser {
    public:
        virtual torch::Tensor step(GradientEvaluator& eval) = 0;
        virtual std::vector<torch::optim::OptimizerParamGroup> param_groups() const = 0;
};
}