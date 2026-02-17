#pragma once
#include "simulation/rng_stream.hpp"
#include <torch/torch.h>

namespace DSO {

struct ThreadContext {
    std::unique_ptr<DSO::RNGStream> rng;

    ThreadContext(std::unique_ptr<DSO::RNGStream> r)
    : rng(std::move(r)) {}

};
} // namespace DSO
