#pragma once
#include <torch/torch.h>
#include "products/product.hpp"
#include "models/stochastic_model.hpp"
#include "control/controller.hpp"
#include "core/time_grid.hpp"

namespace DSO {

struct HedgingResult {
    torch::Tensor terminal_wealth;
    torch::Tensor payoff;
    torch::Tensor pnl;
};

class HedgingEngine {
    public:
        HedgingEngine(
            double initial_cash,
            const ControlIntervals& control_intervals,
            double transaction_cost_rate,
            bool use_smooth_payoff
        ) 
        : control_intervals_(control_intervals) 
        , initial_cash_(initial_cash)
        , transaction_cost_rate_(transaction_cost_rate)
        , use_smooth_payoff_(use_smooth_payoff) {};

        void bind(const SimulationGridSpec& spec) {
            const double T = spec.time_grid.back();
            TORCH_CHECK(control_intervals_.start_times.back() < T - TIME_EPS, "Last control start must be strictly before maturity");
            control_indices_ = bind_control_to_grid(control_intervals_, spec.time_grid);
        };

        HedgingResult run(
            const SimulationResult& simulated,
            const ProductImpl& product,
            ControllerImpl& controller,
            FeatureExtractorImpl& feature_extractor
        ) const {
            const auto& S = simulated.spot;
            const int64_t B = S.size(0);

            torch::Tensor payoff = use_smooth_payoff_ ? product.compute_smooth_payoff(simulated) : product.compute_payoff(simulated);
            torch::Tensor wealth = torch::full({B}, initial_cash_, S.options());
            torch::Tensor position = torch::zeros({B}, S.options());

            SimulationState state;
            state.spot_previous = S.select(1, 0);
            state.spot_cumulative_min = S.select(1, 0);
            state.spot_cumulative_max = S.select(1, 0);
            auto hidden_state_dim = feature_extractor.hidden_state_dim();
            if (hidden_state_dim) state.hidden_state = torch::zeros({B, (int64_t) *hidden_state_dim}, S.options());

            for (size_t k = 0; k < control_intervals_.n_intervals(); ++k) {
                int64_t t0 = control_indices_.start_idx[k];
                int64_t t1 = control_indices_.end_idx[k];

                state.spot = S.select(1, t0);
                state.spot_cumulative_min = torch::minimum(state.spot_cumulative_min, state.spot);
                state.spot_cumulative_max = torch::maximum(state.spot_cumulative_max, state.spot);

                if (simulated.variance.defined()) state.variance = simulated.variance.select(1, t0);
                if (simulated.short_rate.defined()) state.short_rate = simulated.short_rate.select(1, t0);

                state.t = control_intervals_.start_times[k];
                state.t_next = control_intervals_.end_times[k];

                auto fe_output = feature_extractor.forward(state);
                auto hedge = controller.forward(fe_output.features);
                auto trade = hedge - position;
                auto cost = transaction_cost_rate_ * state.spot * torch::abs(trade);
                
                state.hidden_state = fe_output.hidden_state;
                state.spot_previous = state.spot;

                wealth += hedge * (S.select(1, t1) - state.spot) - cost;
                position = hedge;
            }

            torch::Tensor final_spot = S.select(1, control_indices_.end_idx.back());
            wealth -= transaction_cost_rate_ * final_spot * torch::abs(position);

            torch::Tensor pnl = wealth - payoff;
            return {wealth, payoff, pnl};
        }

    private:
        double initial_cash_;
        const ControlIntervals& control_intervals_;
        double transaction_cost_rate_;
        bool use_smooth_payoff_;
        BoundControlIntervals control_indices_;
};

} // namespace DSO