#pragma once
#include "products/product.hpp"

namespace DSO {
class EuropeanCallOption final : public DSO::Product {
    public:
        EuropeanCallOption(double maturity, double strike)
        : maturity_(maturity)
        , strike_(strike) {
            time_grid_.push_back(0.0);
            time_grid_.push_back(maturity);
        };

        const std::vector<double>& time_grid() const override {return time_grid_;}
        const std::vector<FactorType>& factors() const override {return factors_;}

        void compute_payoff(torch::Tensor& paths, torch::Tensor& payoffs) const override {
            auto final_prices = paths.select(-1, -1);
            payoffs = torch::relu(final_prices - strike_);
        };

        const bool include_t0() const override {return false;}
    
    private:
        double maturity_;
        double strike_;
        std::vector<double> time_grid_;
        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};
} // namespace DSO