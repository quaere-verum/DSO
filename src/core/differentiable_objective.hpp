#pragma once
#include <torch/torch.h>
#include "core/threading.hpp"
#include <vector>
#include <string>

namespace DSO {

class DifferentiableObjective {
public:
    virtual ~DifferentiableObjective() = default;
    virtual torch::Tensor loss(
        const torch::Tensor& simulated,
        const BatchSpec& batch,
        const EvalContext& ctx
    ) = 0;
    virtual void bind(const SimulationGridSpec& spec) = 0;
};

} // namespace DSO
