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
        BlackScholesModel(double s0, double r, double sigma, ModelEvalMode mode)
        : s0_(s0)
        , r_(r)
        , sigma_(sigma)
        , mode_(mode) {
            auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            switch (mode) {
                case ModelEvalMode::CALIBRATION: {
                    log_sigma_tensor_ = torch::tensor({(float)std::log(sigma_)}, opt).requires_grad_(true);
                    s0_tensor_ = torch::tensor({(float)s0_}, opt);
                    r_tensor_ = torch::tensor({(float)r_}, opt);
                    parameters_ = {log_sigma_tensor_};
                    param_names_ = {"log_sigma"};
                    use_log_params_ = true;
                    break;
                }
                case ModelEvalMode::VALUATION: {
                    sigma_tensor_ = torch::tensor({(float)sigma_}, opt).requires_grad_(true);
                    s0_tensor_ = torch::tensor({(float)s0_}, opt).requires_grad_(true);
                    r_tensor_ = torch::tensor({(float)r_}, opt).requires_grad_(true);
                    parameters_ = {sigma_tensor_, s0_tensor_, r_tensor_};
                    param_names_ = {"sigma", "s0", "r"};
                    break;
                }
                case ModelEvalMode::HEDGING: {
                    sigma_tensor_ = torch::tensor({(float)sigma_}, opt);
                    s0_tensor_ = torch::tensor({(float)s0_}, opt);
                    r_tensor_ = torch::tensor({(float)r_}, opt);
                    parameters_ = {};
                    param_names_ = {};
                    break;
                }
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
            TORCH_CHECK(init_spec_ != nullptr, "call bind(gridspec) before simulate_batch");
            TORCH_CHECK(mode_ != ModelEvalMode::UNINITIATLISED, "call set_mode(mode) before simulate_batch")
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

            const torch::Tensor sigma = get_sigma_();
            const torch::Tensor drift = (r_tensor_ - 0.5f * sigma * sigma) * dt_;
            const torch::Tensor diff = sigma * sqrt_dt_;

            const torch::Tensor S_future = z.mul_(diff).add_(drift).cumsum_(1).add_(s0_tensor_.log()).exp_();

            if (init_spec_->include_t0) {
                const torch::Tensor S0_col = s0_tensor_.expand({B, 1});
                return torch::cat({S0_col, S_future}, 1);
            }
            return S_future;
        }


        std::vector<torch::Tensor>& parameters() override {
            return parameters_;
        }

        const std::vector<std::string>& parameter_names() const override {
            return param_names_;
        }

        const std::vector<DSO::FactorType>& factors() const override {return factors_;}

        const DSO::ModelEvalMode mode() const override {
            return mode_;
        }

    private:
        torch::Tensor get_sigma_() const {return use_log_params_ ? torch::exp(log_sigma_tensor_) : sigma_tensor_;}

    private:
        double s0_;
        double r_;
        double sigma_;
        bool use_log_params_ = false;
        ModelEvalMode mode_ = ModelEvalMode::UNINITIATLISED;

        torch::Tensor dt_;
        torch::Tensor sqrt_dt_;
        const SimulationGridSpec* init_spec_ = nullptr;

        // parameters
        torch::Tensor s0_tensor_;
        torch::Tensor r_tensor_;
        torch::Tensor sigma_tensor_;
        torch::Tensor log_sigma_tensor_;
        std::vector<torch::Tensor> parameters_;
        std::vector<std::string> param_names_;

        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};

} // namespace DSO
