#pragma once
#include "core/differentiable_objective.hpp"

namespace DSO {
class Optimiser {
    public:
        virtual void step(DSO::DifferentiableObjective& objective) = 0;
};
}