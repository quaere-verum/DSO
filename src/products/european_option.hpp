#pragma once
#include "products/product.hpp"

namespace DSO {
class EuropeanCallOptionImpl final : public OptionImpl {
    public:
        EuropeanCallOptionImpl(double maturity, double strike, double softplus_beta = 1.0)
        : maturity_(maturity)
        , strike_(strike)
        , softplus_beta_(softplus_beta) {
            time_grid_.push_back(maturity_);
            time_indices_ = register_buffer("time_indices", torch::empty({0}, torch::kLong));
        };

        const std::vector<double>& time_grid() const override {return time_grid_;}
        const std::vector<FactorType>& factors() const override {return factors_;}

        torch::Tensor compute_payoff(const SimulationResult& simulated) const override {
            TORCH_CHECK(time_indices_.defined(), "Product: Bind product to SimulationGrid before computing payoff");
            auto final_prices = simulated.spot.index_select(1, time_indices_).squeeze(1);
            return torch::relu(final_prices - strike_);
        };

        torch::Tensor compute_smooth_payoff(const SimulationResult& simulated) const override {
            TORCH_CHECK(time_indices_.defined(), "Product: Bind product to SimulationGrid before computing payoff");
            auto final_prices = simulated.spot.index_select(1, time_indices_).squeeze(1);
            return torch::softplus(final_prices - strike_, softplus_beta_);
        };

        const bool include_t0() const override {return false;}
        const double strike() const override { return strike_; }
        const double maturity() const override { return maturity_; }

        void bind(const SimulationGridSpec& spec) {
            auto time_indices_vec = bind_product_to_grid(time_grid_, spec.time_grid);
            time_indices_ = torch::tensor(time_indices_vec, torch::TensorOptions().dtype(torch::kLong).device(time_indices_.device()));
        }
    
    private:
        double maturity_;
        double strike_;
        double softplus_beta_;
        std::vector<double> time_grid_;
        torch::Tensor time_indices_;
        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};
TORCH_MODULE(EuropeanCallOption);
} // namespace DSO