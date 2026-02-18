#pragma once
#include "simulation/rng_stream.hpp"
#include "core/perf.hpp"
#include <torch/torch.h>

namespace DSO {
struct SimulationGridSpec {
    std::vector<double> time_grid;
    bool include_t0;
};

struct BatchSpec {
    size_t batch_index;
    size_t first_path;
    size_t n_paths;
    uint64_t rng_offset;
};

struct EvalContext {
    std::unique_ptr<DSO::RNGStream> rng;
    torch::Device device = torch::kCPU;
    c10::ScalarType dtype = torch::kFloat32;
    bool training = true;
    PerfCounters* perf = nullptr;

    EvalContext(std::unique_ptr<DSO::RNGStream> r)
    : rng(std::move(r)) {}
};

} // namespace DSO
