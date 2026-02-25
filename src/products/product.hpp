#pragma once
#include <vector>
#include <torch/torch.h>
#include <string>
#include <cstdint>
#include "features/factor.hpp"
#include "simulation/simulation_result.hpp"

namespace DSO {

class Product {
    public:
        virtual const std::vector<double>& time_grid() const = 0;
        virtual const std::vector<FactorType>& factors() const = 0;
        virtual const bool include_t0() const = 0;
        virtual torch::Tensor compute_payoff(const SimulationResult& simulated) const = 0;
        virtual torch::Tensor compute_smooth_payoff(const SimulationResult& simulated) const = 0;

};

class Option : public Product {
    public:
        virtual const double strike() const = 0;
        virtual const double maturity() const = 0;
};
}
