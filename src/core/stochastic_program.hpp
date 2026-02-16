#pragma once
#include "core/differentiable_objective.hpp"

namespace DSO {
class StochasticProgram : public DSO::DifferentiableObjective {
    public:
        virtual void resample_paths(size_t n_paths) = 0;
};
} // namespace DSO