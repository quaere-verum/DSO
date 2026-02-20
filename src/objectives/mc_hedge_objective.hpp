#pragma once
#include <torch/torch.h>
#include "core/stochastic_program.hpp"
#include "products/product.hpp"
#include "control/controller.hpp"
#include "core/threading.hpp"

namespace DSO {
class MCHedgeObjective final : public StochasticProgram {
    public:
        MCHedgeObjective(
            size_t n_paths,
            double initial_cash,
            const Product& product,
            Controller& controller,
            std::vector<double> control_times
        )
        : n_paths_(n_paths)
        , initial_cash_(initial_cash)
        , product_(product)
        , controller_(controller)
        , control_times_(std::move(control_times)) {}

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
            TORCH_CHECK(control_times_.back() < T, "MCHedgeObjective: ctrl index out of range for simulated");

            torch::Tensor payoff = torch::empty({B}, simulated.options().dtype(torch::kFloat32));
            product_.compute_payoff(simulated, payoff);

            torch::Tensor value = torch::full({B}, initial_cash_, simulated.options().dtype(torch::kFloat32));

            for (size_t k = 0; k + 1 < control_indices_.size(); ++k) {
                const int64_t t0_idx = control_indices_[k];
                const int64_t t1_idx = control_indices_[k + 1];

                // spot at decision time
                torch::Tensor S0 = simulated.select(1, t0_idx);
                torch::Tensor S1 = simulated.select(1, t1_idx);
                MarketView mv;
                mv.S_t = S0;
                mv.t = control_times_[k];
                mv.t_next = control_times_[k+1];
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
            control_indices_.clear();
            control_indices_.reserve(control_times_.size());

            for (double t : control_times_) {
                int64_t idx = find_time_index_(spec.time_grid, t);
                TORCH_CHECK(idx >= 0, "control time not in simulation grid");
                control_indices_.push_back(idx);
            }
            TORCH_CHECK(control_indices_.back() < (int64_t)spec.time_grid.size(), "index out of range");
            bound_ = true;
        }
    private:
        static int64_t find_time_index_(const std::vector<double>& grid, double t) {
            constexpr double eps = 1e-12;
            auto it = std::lower_bound(grid.begin(), grid.end(), t - eps);
            if (it == grid.end()) return -1;
            if (std::abs(*it - t) > eps) return -1;
            return (int64_t)std::distance(grid.begin(), it);
        }
    private:
        size_t n_paths_;
        double initial_cash_;
        const Product& product_;
        Controller& controller_;
        std::vector<double> control_times_;
        std::vector<int64_t> control_indices_;
        bool bound_ = false;

        size_t epoch_ = 0;
        uint64_t epoch_rng_offset_ = 0;
};

} // namespace DSO
