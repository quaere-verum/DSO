#pragma once
#include <torch/torch.h>
#include "core/stochastic_program.hpp"
#include "products/product.hpp"
#include "control/controller.hpp"
#include "core/threading.hpp"
#include "core/time_grid.hpp"

namespace DSO {
class MCHedgeObjective final : public StochasticProgram {
    public:
        MCHedgeObjective(
            size_t n_paths,
            double initial_cash,
            const Product& product,
            Controller& controller,
            const ControlIntervals& control_intervals
        )
        : n_paths_(n_paths)
        , initial_cash_(initial_cash)
        , product_(product)
        , controller_(controller)
        , control_intervals_(control_intervals) {
            control_intervals_.validate(TIME_EPS);
        }

        torch::Tensor loss(
            const torch::Tensor& simulated,
            const BatchSpec& batch,
            const EvalContext& ctx
        ) override {
            TORCH_CHECK(simulated.defined(), "MCHedgeObjective: simulated must be defined");
            TORCH_CHECK(simulated.dim() == 2, "MCHedgeObjective: simulated must be 2D [B,T]");
            const int64_t B = simulated.size(0);
            const int64_t T = simulated.size(1);
            TORCH_CHECK(B > 0 && T > 1, "MCHedgeObjective: simulated must have shape [B, T] with T>1");

            torch::Tensor payoff = torch::empty({B}, simulated.options().dtype(torch::kFloat32));
            product_.compute_payoff(simulated, payoff);

            torch::Tensor value = torch::full({B}, (float)initial_cash_, simulated.options().dtype(torch::kFloat32));

            for (size_t k = 0; k < control_intervals_.n_intervals(); ++k) {
                const int64_t t0_idx = control_indices_.start_idx[k];
                const int64_t t1_idx = control_indices_.end_idx[k];

                // spot at decision time
                torch::Tensor S0 = simulated.select(1, t0_idx);
                torch::Tensor S1 = simulated.select(1, t1_idx);
                MarketView mv;
                mv.S_t = S0;
                mv.t = control_intervals_.start_times[k];
                mv.t_next = control_intervals_.end_times[k];
                mv.t_index = k;

                torch::Tensor hedge = controller_.action(mv, product_, batch, ctx);

                if (hedge.dim() == 2) {
                    TORCH_CHECK(hedge.size(1) == 1, "MCHedgeObjective: controller action [B,1] expected if 2D");
                    hedge = hedge.squeeze(1);
                }
                TORCH_CHECK(hedge.sizes() == torch::IntArrayRef({B}), "MCHedgeObjective: controller action must be [B] or [B,1]");

                value = value + hedge * (S1 - S0);
            }
            return value.sub_(payoff).square_().mean();
        }

        void resample_paths(size_t n_paths) override {
            n_paths_ = n_paths;
            ++epoch_;
            epoch_rng_offset_ = static_cast<uint64_t>(epoch_) * (1ULL << 32);
        }

        size_t n_paths() const { return n_paths_; }
        uint64_t epoch_rng_offset() const { return epoch_rng_offset_; }
        void bind(const SimulationGridSpec& spec) override {
            const double T = spec.time_grid.back();
            TORCH_CHECK(control_intervals_.start_times.back() < T - TIME_EPS, "Last control start must be strictly before maturity");
            control_indices_ = bind_to_grid(control_intervals_, spec.time_grid);
        }

    private:
        size_t n_paths_;
        double initial_cash_;
        const Product& product_;
        Controller& controller_;
        const ControlIntervals& control_intervals_;
        BoundControlIntervals control_indices_;
        bool bound_ = false;

        size_t epoch_ = 0;
        uint64_t epoch_rng_offset_ = 0;
};

} // namespace DSO
