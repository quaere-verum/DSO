#pragma once
#include <torch/torch.h>
#include "core/threading.hpp"
#include "models/stochastic_model.hpp"
#include <vector>
#include <string>

namespace DSO {

class DifferentiableObjective {
public:
    virtual ~DifferentiableObjective() = default;
    virtual torch::Tensor loss(
        const SimulationResult& simulated,
        const BatchSpec& batch,
        const EvalContext& ctx
    ) = 0;
};

} // namespace DSO
