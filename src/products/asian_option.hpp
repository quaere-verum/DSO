#pragma once
#include "products/product.hpp"
#include <iostream>
#include <tuple>

namespace DSO {
class AsianCallOption final : public DSO::Product {
    public:
        AsianCallOption(double maturity, double strike, size_t n_steps)
        : strike_(strike) {
            double dt = maturity / n_steps;
            for (size_t i = 0; i <= n_steps; ++i) {
                time_grid_.push_back(i * dt);
            }
        }

        const std::vector<double>& time_grid() const override { return time_grid_; }
        const std::vector<FactorType>& factors() const override { return factors_; }

        void compute_payoff(torch::Tensor& paths, torch::Tensor& payoffs) const override {
            auto average_prices = paths.mean(1);
            payoffs = torch::relu(average_prices - strike_);
        }

        const bool include_t0() const override { return true; }
    
    private:
        double strike_;
        std::vector<double> time_grid_;
        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};
} // namespace DSO