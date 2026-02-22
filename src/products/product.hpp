#pragma once
#include <vector>
#include <torch/torch.h>
#include <string>
#include <cstdint>

namespace DSO {
enum class FactorType : uint8_t {
    Spot = 0,
    LogSpot = 1,
    Numeraire = 2,
    DiscountFactor = 3,
    ShortRate = 4,
    Variance = 5,
    FXRate = 6,
    // Path functionals:
    RunningAverage = 7,
    RunningMax = 8,
    RunningMin = 9,
    BarrierHit = 10
};

struct FactorSpec {
    std::vector<FactorType> factors;
    size_t n_factors() const { return factors.size(); }
};

class Product {
    public:
        virtual const std::vector<double>& time_grid() const = 0;
        virtual const std::vector<FactorType>& factors() const = 0;
        virtual const bool include_t0() const = 0;
        virtual void compute_payoff(const torch::Tensor& paths, torch::Tensor& payoffs) const = 0;
        virtual void compute_smooth_payoff(const torch::Tensor& paths, torch::Tensor& payoffs) const = 0;

};

class Option : public Product {
    public:
        virtual const double strike() const = 0;
        virtual const double maturity() const = 0;
};
}
