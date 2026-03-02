#pragma once
#include "products/product.hpp"
#include "core/threading.hpp"
#include "core/time_grid.hpp"
#include <iostream>
#include <tuple>

namespace DSO {
class LookbackCallOptionImpl final : public OptionImpl {
    public:
        LookbackCallOptionImpl(double maturity, double strike, std::vector<double> monitoring_grid, double softplus_beta = 1.0)
        : strike_(strike)
        , maturity_(maturity)
        , time_grid_(std::move(monitoring_grid))
        , softplus_beta_(softplus_beta) {
            time_indices_ = register_buffer("time_indices", torch::empty({0}, torch::kLong));
        }

        const ProductTimes& time_grid() const override { return time_grid_; }
        const std::vector<FactorType>& factors() const override { return factors_; }

        torch::Tensor compute_payoff(const SimulationResult& simulated) const override {
            TORCH_CHECK(time_indices_.defined(), "Product: Bind product to SimulationGrid before computing payoff");
            auto max_prices = std::get<0>(simulated.spot.index_select(1, time_indices_).max(1));
            return torch::relu(max_prices - strike_);
        }

        torch::Tensor compute_smooth_payoff(const SimulationResult& simulated) const override {
            TORCH_CHECK(time_indices_.defined(), "Product: Bind product to SimulationGrid before computing payoff");
            auto max_prices = std::get<0>(simulated.spot.index_select(1, time_indices_).max(1));
            return torch::softplus(max_prices - strike_, softplus_beta_);
        };

        void bind(const SimulationGridSpec& spec) {
            auto time_indices_vec = bind_product_to_grid(time_grid_, spec.time_grid);
            time_indices_ = torch::tensor(time_indices_vec, torch::TensorOptions().dtype(torch::kLong).device(time_indices_.device()));
        }

        const bool include_t0() const override { return false; }
        const double strike() const override { return strike_; }
        const double maturity() const override { return maturity_; }        
    
    private:
        double strike_;
        double maturity_;
        double softplus_beta_;
        ProductTimes time_grid_;
        torch::Tensor time_indices_;
        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};
TORCH_MODULE(LookbackCallOption);
} // namespace DSO