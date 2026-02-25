#pragma once
#include <torch/torch.h>

namespace DSO {
struct SimulationResult {
    size_t n_paths;
    torch::Tensor spot;
    torch::Tensor variance;
    torch::Tensor short_rate;
};
} // namespace DSO