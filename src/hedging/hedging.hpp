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
            const ControlIntervals& control_intervals
        ) 
        : control_intervals_(control_intervals) 
        , initial_cash_(initial_cash) {

        };

        void bind(const SimulationGridSpec& spec) {
            const double T = spec.time_grid.back();
            TORCH_CHECK(control_intervals_.start_times.back() < T - TIME_EPS, "Last control start must be strictly before maturity");
            control_indices_ = bind_to_grid(control_intervals_, spec.time_grid);
        };

        HedgingResult run(
            const SimulationResult& simulated,
            const Product& product,
            const ControllerImpl& controller
        ) const {
            const auto& S = simulated.spot;
            const int64_t B = S.size(0);

            torch::Tensor payoff = product.compute_payoff(simulated);
            torch::Tensor wealth = torch::full({B}, initial_cash_, S.options());

            MarketView state;

            for (size_t k = 0; k < control_intervals_.n_intervals(); ++k) {

                int64_t t0 = control_indices_.start_idx[k];
                int64_t t1 = control_indices_.end_idx[k];

                state.spot = S.select(1, t0);

                if (simulated.variance.defined()) state.variance = simulated.variance.select(1, t0);
                if (simulated.short_rate.defined()) state.short_rate = simulated.short_rate.select(1, t0);

                state.t = control_intervals_.start_times[k];
                state.t_next = control_intervals_.end_times[k];

                torch::Tensor hedge = controller.forward(state).squeeze(-1);

                wealth += hedge * (S.select(1, t1) - state.spot);
            }

            torch::Tensor pnl = wealth - payoff;

            return {wealth, payoff, pnl};
        }

    private:
        double initial_cash_;
        const ControlIntervals& control_intervals_;
        BoundControlIntervals control_indices_;
};

} // namespace DSO