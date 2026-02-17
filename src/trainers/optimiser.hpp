#pragma once
#include "core/differentiable_objective.hpp"

namespace DSO {
class Optimiser {
    public:
        virtual torch::Tensor step(DSO::DifferentiableObjective& objective) = 0;
};
}