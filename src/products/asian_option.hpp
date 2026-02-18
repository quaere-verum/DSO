#pragma once
#include "products/product.hpp"
#include <iostream>
#include <tuple>

namespace DSO {
class AsianCallOption final : public Option {
    public:
        AsianCallOption(double maturity, double strike, size_t n_steps)
        : strike_(strike)
        , maturity_(maturity) {
            double dt = maturity / n_steps;
            for (size_t i = 0; i <= n_steps; ++i) {
                time_grid_.push_back(i * dt);
            }
        }

        const std::vector<double>& time_grid() const override { return time_grid_; }
        const std::vector<FactorType>& factors() const override { return factors_; }

        void compute_payoff(const torch::Tensor& paths, torch::Tensor& payoffs) const override {
            auto average_prices = paths.mean(1);
            payoffs = torch::relu(average_prices - strike_);
        }

        const bool include_t0() const override { return true; }
        const double strike() const override { return strike_; }
        const double maturity() const override { return maturity_; }
    
    private:
        double strike_;
        double maturity_;
        std::vector<double> time_grid_;
        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};
} // namespace DSO