#pragma once
#include "products/product.hpp"
#include "core/threading.hpp"
#include "core/time_grid.hpp"
#include <iostream>
#include <tuple>

namespace DSO {

class DownAndOutCallOptionImpl final : public ProductImpl {
    public:
        DownAndOutCallOptionImpl(
            double maturity,
            double strike,
            double barrier,
            std::vector<double> monitoring_grid,
            double softplus_beta = 1.0,
            double barrier_beta = 20.0
        )
        : maturity_(maturity)
        , strike_(strike)
        , barrier_(barrier)
        , time_grid_(std::move(monitoring_grid))
        , softplus_beta_(softplus_beta)
        , barrier_beta_(barrier_beta) {
            time_indices_ = register_buffer("time_indices", torch::empty({0}, torch::kLong));
        }

        const ProductTimes& time_grid() const override { return time_grid_; }
        const std::vector<FactorType>& factors() const override { return factors_; }

        torch::Tensor compute_payoff(const SimulationResult& simulated) const override {
            TORCH_CHECK(time_indices_.defined(), "Product: Bind product to SimulationGrid before computing payoff");

            auto monitored_spot = simulated.spot.index_select(1, time_indices_);
            auto min_prices = std::get<0>(monitored_spot.min(1));

            auto final_price = simulated.spot.select(1, maturity_index_);

            auto intrinsic = torch::relu(final_price - strike_);

            auto alive = (min_prices > barrier_).to(intrinsic.dtype());

            return intrinsic * alive;
        }

        torch::Tensor compute_smooth_payoff(const SimulationResult& simulated) const override {
            TORCH_CHECK(time_indices_.defined(), "Product: Bind product to SimulationGrid before computing payoff");

            auto monitored_spot = simulated.spot.index_select(1, time_indices_);
            auto min_prices = std::get<0>(monitored_spot.min(1));

            auto final_price = simulated.spot.select(1, maturity_index_);

            auto intrinsic = torch::softplus(final_price - strike_, softplus_beta_);

            auto alive_prob = torch::sigmoid(barrier_beta_ * (min_prices - barrier_));

            return intrinsic * alive_prob;
        }

        void bind(const SimulationGridSpec& spec) {
            auto time_indices_vec = bind_product_to_grid(time_grid_, spec.time_grid);
            time_indices_ = torch::tensor(
                time_indices_vec,
                torch::TensorOptions().dtype(torch::kLong).device(time_indices_.device())
            );

            maturity_index_ = time_indices_.select(0, -1).item<int64_t>();
        }

        const bool include_t0() const override { return false; }

        const double maturity() const { return maturity_; }
        const double barrier() const { return barrier_; }
        const double strike() const { return strike_; }

    private:
        double maturity_;
        double strike_;
        double barrier_;
        double softplus_beta_;
        double barrier_beta_;

        ProductTimes time_grid_;

        torch::Tensor time_indices_;
        int64_t maturity_index_;

        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};

TORCH_MODULE(DownAndOutCallOption);

} // namespace DSO