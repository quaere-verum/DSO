#pragma once
#include "simulation/rng_stream.hpp"
#include <torch/torch.h>

namespace DSO {

struct ThreadContext {
    std::unique_ptr<DSO::RNGStream> rng;
    torch::Tensor z;
    torch::Device device = torch::kCPU;

    ThreadContext(std::unique_ptr<DSO::RNGStream> r)
    : rng(std::move(r)) {}

    void ensure_tensor_(
        torch::IntArrayRef sizes
    ) {
        if (!z.defined() || z.sizes() != sizes || !z.is_contiguous()) {
            z = torch::empty(sizes, torch::TensorOptions().dtype(torch::kFloat32).device(device));
        }
    }
};
} // namespace DSO
