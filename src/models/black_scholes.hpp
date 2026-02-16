#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <memory>

#include "models/stochastic_model.hpp"
#include "core/threading.hpp"
#include "products/product.hpp"
#include "simulation/monte_carlo.hpp"

namespace DSO {

class BlackScholesModel final : public StochasticModel {
    public:
        struct Config {
            MonteCarloExecutor::Config monte_carlo;
            bool use_log_params = true;
        };

        BlackScholesModel(double s0, double r, double sigma, Config config)
        : config_(std::move(config))
        , monte_carlo_engine_(std::make_unique<MonteCarloExecutor>(config_.monte_carlo)) {

            auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

            s0_ = torch::tensor({s0}, opt);
            r_ = torch::tensor({r}, opt);

            if (config_.use_log_params) {
                log_sig_ = torch::tensor({(float)std::log(sigma)}, opt).requires_grad_(true);
                param_names_ = {"log_sigma"};
            } else {
                sig_ = torch::tensor({(float)sigma}, opt).requires_grad_(true);
                param_names_ = {"sigma"};
            }
        }

        void init(const DSO::Product& product) {
            const std::vector<double>& time_grid = product.time_grid();
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
            init_product_ = &product;
        }

        torch::Tensor simulate_paths(
            size_t n_paths,
            const Product& product,
            uint64_t rng_stream_offset
        ) override {
            TORCH_CHECK(n_paths > 0, "n_paths must be > 0");
            TORCH_CHECK(init_product_ != nullptr, "call init(product) before simulate_paths");
            TORCH_CHECK(&product == init_product_, "simulate_paths called with a different product than init()");

            // We assume CPU-only RNGStream.
            // If you later add GPU RNG, this can be generalized.
            const auto device = torch::kCPU;

            // Product provides time_grid. Assume it includes t0 and is increasing.
            const int64_t n_steps = dt_.numel();

            const torch::Tensor sig = get_sigma_();
            const torch::Tensor drift = (r_ - 0.5f * sig * sig) * dt_;
            const torch::Tensor diff = sig * sqrt_dt_;
            const torch::Tensor log_s0 = torch::log(s0_);

            // We will generate Z ~ N(0,1) for each path and step: [n_paths, n_steps] (no grad)
            // Then compute logS via torch ops (grad flows through sigma).
            struct Parts {
                std::vector<torch::Tensor> chunks; // each chunk is [batch_n, n_times]
                Parts() = default;
                Parts(Parts&&) = default;
                Parts& operator=(Parts&&) = default;
                void merge(Parts& other) { // MergeableResult requirement (accept either void or R&)
                    if (!other.chunks.empty()) {
                        chunks.reserve(chunks.size() + other.chunks.size());
                        for (auto& t : other.chunks) chunks.emplace_back(std::move(t));
                        other.chunks.clear();
                    }
                }
            };
            auto parts = monte_carlo_engine_->run<Parts>(n_paths,
                [&](size_t /*b*/, size_t first_path, size_t batch_n, DSO::ThreadContext& ctx) -> Parts {

                    TORCH_CHECK(ctx.rng.get() != nullptr, "ThreadContext.rng must be set");
                    TORCH_CHECK(ctx.device.is_cpu(), "BlackScholesModel: CPU only");

                    const int64_t B = static_cast<int64_t>(batch_n);
                    ctx.ensure_tensor_({B, n_steps});

                    float* z_ptr = ctx.z.data_ptr<float>();

                    for (int64_t i = 0; i < B; ++i) {
                        const uint64_t path_idx = static_cast<uint64_t>(first_path) + static_cast<uint64_t>(i);
                        ctx.rng->seed_path(path_idx + rng_stream_offset);
                        ctx.rng->fill_normal(z_ptr + i * n_steps, n_steps, 0.0, 1.0);
                    }

                    const torch::Tensor increments = drift + ctx.z * diff;

                    const torch::Tensor log_paths = log_s0 + torch::cumsum(increments, /*dim=*/1);
                    const torch::Tensor S_future = torch::exp(log_paths);

                    torch::Tensor paths;
                    // prepend initial spot column
                    if (product.include_t0()) {
                        const torch::Tensor S0_col = s0_.expand({B, 1});
                        paths = torch::cat({S0_col, S_future}, /*dim=*/1);
                    } else {
                        paths = S_future;
                    }

                    Parts out;
                    out.chunks.emplace_back(paths);
                    return out;
                }
            );

            // Deterministic assembly in batch order
            TORCH_CHECK(!parts.chunks.empty(), "internal error: no chunks returned");

            // One final cat: differentiable and preserves grad to params
            torch::Tensor all_paths = torch::cat(parts.chunks, /*dim=*/0);
            const int64_t expected_times = product.include_t0() ? (n_steps + 1) : n_steps;
            TORCH_CHECK(all_paths.sizes() == torch::IntArrayRef({(int64_t)n_paths, expected_times}),
                        "internal error: assembled paths have wrong shape");

            return all_paths;
        }

        std::vector<torch::Tensor> parameters() override {
            if (config_.use_log_params) return {log_sig_};
            return {sig_};
        }

        const std::vector<std::string>& parameter_names() const override {
            return param_names_;
        }

    private:
        torch::Tensor get_sigma_() const {
            return config_.use_log_params ? torch::exp(log_sig_) : sig_;
        }

    private:
        Config config_;
        std::unique_ptr<MonteCarloExecutor> monte_carlo_engine_;

        torch::Tensor dt_;
        torch::Tensor sqrt_dt_;
        const DSO::Product* init_product_ = nullptr;

        // parameters
        torch::Tensor s0_;
        torch::Tensor r_;
        torch::Tensor sig_;
        torch::Tensor log_sig_;
        std::vector<std::string> param_names_;
};

} // namespace DSO
