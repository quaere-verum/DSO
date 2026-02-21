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

class BlackScholesModel final : public StochasticModel {
    public:
        struct Config {
            bool use_log_params = true;
        };

        BlackScholesModel(double s0, double r, double sigma, Config config)
        : config_(std::move(config)) {
            auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            s0_ = torch::tensor({s0}, opt);
            r_ = torch::tensor({r}, opt);

            if (config_.use_log_params) {
                log_sigma_ = torch::tensor({(float)std::log(sigma)}, opt).requires_grad_(true);
                param_names_ = {"log_sigma"};
            } else {
                sigma_ = torch::tensor({(float)sigma}, opt).requires_grad_(true);
                param_names_ = {"sigma"};
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

        torch::Tensor simulate_batch(
            const BatchSpec& batch,
            const EvalContext& ctx,
            Controller*
        ) override {
            TORCH_CHECK(batch.n_paths > 0, "batch.n_paths must be > 0");
            TORCH_CHECK(init_spec_ != nullptr, "call init(product) before simulate_batch");
            TORCH_CHECK(ctx.rng.get() != nullptr, "ThreadContext.rng must be set");

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

            const torch::Tensor sig = get_sigma_();
            const torch::Tensor drift = (r_ - 0.5f * sig * sig) * dt_;
            const torch::Tensor diff = sig * sqrt_dt_;

            const torch::Tensor S_future = z.mul_(diff).add_(drift).cumsum_(1).add_(s0_.log()).exp_();

            if (init_spec_->include_t0) {
                const torch::Tensor S0_col = s0_.expand({B, 1});
                return torch::cat({S0_col, S_future}, 1);
            }
            return S_future;
        }


        std::vector<torch::Tensor> parameters() override {
            if (config_.use_log_params) return {log_sigma_};
            return {sigma_};
        }

        const std::vector<std::string>& parameter_names() const override {
            return param_names_;
        }

        const std::vector<DSO::FactorType>& factors() const override {return factors_;}

        void set_training(bool training) {
            if (config_.use_log_params) {
                log_sigma_.requires_grad_(training);
            } else {
                sigma_.requires_grad_(training);
            }
        }

    private:
        torch::Tensor get_sigma_() const {return config_.use_log_params ? torch::exp(log_sigma_) : sigma_;}

        static std::vector<double> merge_time_grids_(
            const std::vector<double>& a,
            const std::vector<double>& b,
            double eps = 1e-12
        ) {
            std::vector<double> out;
            out.reserve(a.size() + b.size());
            auto push_unique = [&](double t) {
                if (out.empty() || std::abs(out.back() - t) > eps) out.push_back(t);
            };

            size_t i = 0, j = 0;
            while (i < a.size() || j < b.size()) {
                double ta = (i < a.size()) ? a[i] : std::numeric_limits<double>::infinity();
                double tb = (j < b.size()) ? b[j] : std::numeric_limits<double>::infinity();
                if (ta <= tb + eps) { push_unique(ta); ++i; if (std::abs(ta - tb) <= eps) ++j; }
                else { push_unique(tb); ++j; }
            }
            return out;
        }

    private:
        Config config_;

        torch::Tensor dt_;
        torch::Tensor sqrt_dt_;
        const SimulationGridSpec* init_spec_ = nullptr;

        // parameters
        torch::Tensor s0_;
        torch::Tensor r_;
        torch::Tensor sigma_;
        torch::Tensor log_sigma_;
        std::vector<std::string> param_names_;

        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};

} // namespace DSO
