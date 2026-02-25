#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>

#include "models/stochastic_model.hpp"
#include "core/threading.hpp"
#include "products/product.hpp"
#include "simulation/monte_carlo.hpp"

namespace DSO {

class BlackScholesModelImpl final : public StochasticModelImpl {
    public:
        BlackScholesModelImpl(double s0, double sigma, bool use_log_sigma)
        : use_log_sigma_(use_log_sigma) {
            auto opt = torch::TensorOptions().dtype(torch::kFloat32);
            s0_tensor_ = register_parameter("s0", torch::tensor({static_cast<float>(s0)}, opt));
            if (use_log_sigma_) {
                log_sigma_tensor_ = register_parameter("log_sigma", torch::tensor({static_cast<float>(std::log(sigma))}, opt));
            } else {
                sigma_tensor_ = register_parameter("sigma", torch::tensor({static_cast<float>(sigma)}, opt));
            }
            
        }

        void init(const SimulationGridSpec& spec) override {
            const auto& time_grid = spec.time_grid;
            const int64_t n_times = static_cast<int64_t>(time_grid.size());
            const int64_t n_steps = n_times - 1;

            std::vector<float> dt_host(n_steps);
            for (int64_t j = 0; j < n_steps; ++j) {
                double dt = time_grid[j + 1] - time_grid[j];
                TORCH_CHECK(dt > 0, "dt must be > 0 for all steps");
                dt_host[j] = static_cast<float>(dt);
            }
            dt_ = torch::from_blob(dt_host.data(), {n_steps}, torch::TensorOptions().dtype(torch::kFloat32)).clone();
            sqrt_dt_ = torch::sqrt(dt_);
            init_spec_ = &spec;
        }

        SimulationResult simulate_batch(
            const BatchSpec& batch,
            const EvalContext& ctx,
            std::shared_ptr<ControllerImpl>
        ) override {
            TORCH_CHECK(batch.n_paths > 0, "batch.n_paths must be > 0");
            TORCH_CHECK(init_spec_ != nullptr, "call bind(gridspec) before simulate_batch");
            TORCH_CHECK(ctx.rng.get() != nullptr, "ThreadContext.rng must be set");

            SimulationResult out;
            const auto device = torch::kCPU;
            const int64_t n_steps = dt_.numel();
            const int64_t B = static_cast<int64_t>(batch.n_paths);

            auto z = torch::empty({B, n_steps}, torch::TensorOptions().device(torch::kCPU).dtype(torch::kFloat32));

            float* z_ptr = z.data_ptr<float>();

            auto* perf = ctx.perf;
            {
                std::optional<DSO::ScopedTimer> rng_timer;
                if (perf) rng_timer.emplace(DSO::ScopedTimer(*perf, DSO::Stage::Rng));

                for (int64_t i = 0; i < B; ++i) {
                    const uint64_t path_idx = static_cast<uint64_t>(batch.first_path) + static_cast<uint64_t>(i);
                    ctx.rng->seed_path(path_idx + batch.rng_offset);
                    ctx.rng->fill_normal(z_ptr + i * n_steps, n_steps, 0.0, 1.0);
                }
            }
            

            std::optional<DSO::ScopedTimer> timer;
            if (perf) timer.emplace(*perf, DSO::Stage::SimMath);

            // Simulate the forward price, P(0, T) measure
            const torch::Tensor sigma = get_sigma_();
            const torch::Tensor drift = -0.5f * sigma * sigma * dt_;
            const torch::Tensor diff = sigma * sqrt_dt_;

            const torch::Tensor S_future = z.mul_(diff).add_(drift).cumsum_(1).add_(s0_tensor_.log()).exp_();

            if (init_spec_->include_t0) {
                const torch::Tensor S0_col = s0_tensor_.expand({B, 1});
                out.spot = torch::cat({S0_col, S_future}, 1);
            } else {
                out.spot = S_future;
            }
            return out;
        }

        const std::vector<DSO::FactorType>& factors() const override {return factors_;}

    private:
        torch::Tensor get_sigma_() const { return use_log_sigma_ ? torch::exp(log_sigma_tensor_) : sigma_tensor_; }

    private:
        bool use_log_sigma_;

        torch::Tensor dt_;
        torch::Tensor sqrt_dt_;
        const SimulationGridSpec* init_spec_ = nullptr;

        // parameters
        torch::Tensor s0_tensor_;
        torch::Tensor log_sigma_tensor_;
        torch::Tensor sigma_tensor_;

        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};
TORCH_MODULE(BlackScholesModel);
} // namespace DSO
