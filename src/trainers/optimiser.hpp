#pragma once
#include "core/differentiable_objective.hpp"

namespace DSO {
class GradientEvaluator {
    public:
        virtual ~GradientEvaluator() = default;
        virtual torch::Tensor evaluate_and_set_grads() = 0;
};
class Optimiser {
    public:
        virtual torch::Tensor step(GradientEvaluator& eval) = 0;
};
}