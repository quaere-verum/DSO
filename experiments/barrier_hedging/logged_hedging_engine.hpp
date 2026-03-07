#pragma once
#include <torch/torch.h>
#include "dso.hpp"

struct LoggedHedgingResult {
    torch::Tensor terminal_wealth;
    torch::Tensor payoff;
    torch::Tensor pnl;

    torch::Tensor total_transaction_cost;
    torch::Tensor total_turnover;

    torch::Tensor hedges;
    torch::Tensor wealth_path;
};

class LoggedHedgingEngine {
public:
    LoggedHedgingEngine(
        double initial_cash,
        const DSO::ControlIntervals& control_intervals,
        double transaction_cost_rate
    )
        : control_intervals_(control_intervals),
          initial_cash_(initial_cash),
          transaction_cost_rate_(transaction_cost_rate) {}

    void bind(const DSO::SimulationGridSpec& spec) {
        const double T = spec.time_grid.back();

        TORCH_CHECK(
            control_intervals_.start_times.back() < T - 1e-12,
            "Last control start must be strictly before maturity"
        );

        control_indices_ =
            DSO::bind_control_to_grid(control_intervals_, spec.time_grid);
    }

    LoggedHedgingResult run(
        const DSO::SimulationResult& simulated,
        const DSO::ProductImpl& product,
        DSO::ControllerImpl& controller,
        DSO::FeatureExtractorImpl& feature_extractor
    ) const {

        const auto& S = simulated.spot;
        const int64_t B = S.size(0);
        const int64_t N = control_intervals_.n_intervals();

        auto options = S.options();

        LoggedHedgingResult out;

        out.payoff = product.compute_payoff(simulated); 

        torch::Tensor wealth = torch::full({B}, initial_cash_, options);
        torch::Tensor position = torch::zeros({B}, options);

        out.hedges = torch::zeros({B, N}, options);
        out.wealth_path = torch::zeros({B, N + 1}, options);

        torch::Tensor total_cost = torch::zeros({B}, options);
        torch::Tensor total_turnover = torch::zeros({B}, options);

        out.wealth_path.select(1, 0).copy_(wealth);

        DSO::SimulationState state;
        state.spot_previous = S.select(1, 0);
        state.spot_cumulative_min = S.select(1, 0);
        state.spot_cumulative_max = S.select(1, 0);

        auto hidden_dim = feature_extractor.hidden_state_dim();
        if (hidden_dim) {
            state.hidden_state = torch::zeros({B, static_cast<int64_t>(*hidden_dim)}, options);
        }

        for (int64_t k = 0; k < N; ++k) {

            int64_t t0 = control_indices_.start_idx[k];
            int64_t t1 = control_indices_.end_idx[k];

            state.spot = S.select(1, t0);
            state.spot_cumulative_min = torch::minimum(state.spot_cumulative_min, state.spot);
            state.spot_cumulative_max = torch::maximum(state.spot_cumulative_max, state.spot);

            if (simulated.variance.defined())
                state.variance = simulated.variance.select(1, t0);

            if (simulated.short_rate.defined())
                state.short_rate = simulated.short_rate.select(1, t0);

            state.t = control_intervals_.start_times[k];
            state.t_next = control_intervals_.end_times[k];

            auto fe_out = feature_extractor.forward(state);

            torch::Tensor hedge = controller.forward(fe_out.features);

            out.hedges.select(1, k).copy_(hedge);

            torch::Tensor trade = hedge - position;

            torch::Tensor cost =
                transaction_cost_rate_ *
                state.spot *
                torch::abs(trade);

            total_cost += cost;
            total_turnover += torch::abs(trade);

            torch::Tensor dS = S.select(1, t1) - state.spot;

            wealth += hedge * dS - cost;

            position = hedge;
            state.hidden_state = fe_out.hidden_state;
            state.spot_previous = state.spot;

            out.wealth_path.select(1, k + 1).copy_(wealth);
        }

        torch::Tensor final_spot =
            S.select(1, control_indices_.end_idx.back());

        torch::Tensor final_cost =
            transaction_cost_rate_ *
            final_spot *
            torch::abs(position);

        wealth -= final_cost;
        total_cost += final_cost;
        total_turnover += torch::abs(position);

        out.terminal_wealth = wealth;
        out.pnl = wealth - out.payoff;

        out.total_transaction_cost = total_cost;
        out.total_turnover = total_turnover;

        return out;
    }

private:
    double initial_cash_;
    const DSO::ControlIntervals& control_intervals_;
    double transaction_cost_rate_;

    DSO::BoundControlIntervals control_indices_;
};