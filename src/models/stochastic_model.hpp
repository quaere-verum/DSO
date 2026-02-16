#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include "core/threading.hpp"
#include "products/product.hpp"
#include "simulation/monte_carlo.hpp"

namespace DSO {
class StochasticModel {
    public:
        virtual torch::Tensor simulate_paths(
            size_t n_paths,
            const Product& product,
            uint64_t rng_stream_offset
        ) = 0;
        virtual std::vector<torch::Tensor> parameters() = 0;
        virtual const std::vector<std::string>& parameter_names() const = 0;
};
} // namespace DSO