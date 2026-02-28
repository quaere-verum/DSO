#pragma once
#include "core/differentiable_objective.hpp"

namespace DSO {
class StochasticProgram : public DSO::DifferentiableObjective {
    public:
        virtual void resample_paths() = 0;
        virtual uint64_t epoch_rng_offset() const = 0;
};
} // namespace DSO