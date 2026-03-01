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

struct BlackScholesModelParameters {
    double s0;
    double sigma;
};

torch::Tensor compute_black_scholes_price(
    torch::Tensor s0, 
    torch::Tensor sigma,
    torch::Tensor K, 
    double T, 
    double r = 0.0
) {
    auto r_tensor = torch::tensor(r, s0.options());
    auto T_sqrt = std::sqrt(T);
    auto d1 = (torch::log(s0 / K) + (r + 0.5 * sigma.pow(2)) * T) / (sigma * T_sqrt);
    auto d2 = d1 - sigma * T_sqrt;
    auto price = s0 * torch::special::ndtr(d1) - K * torch::exp(-r_tensor * T) * torch::special::ndtr(d2);

    return price;
}

BlackScholesModelParameters calibrate_black_scholes_model_parameters(
    double market_price, 
    double market_strike,
    double s0,
    double maturity,
    bool verbose = false
) {
    auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    auto market_price_scaled = torch::tensor(market_price / s0, opt);
    auto market_strike_scaled = torch::tensor(market_strike / s0, opt);
    
    auto s0_tensor = torch::tensor(1.0, opt);
    auto raw_sigma = torch::log(torch::tensor(0.20, opt)).set_requires_grad(true);

    std::vector<torch::Tensor> params = {raw_sigma};
    auto optim_options = torch::optim::LBFGSOptions().lr(1.0).max_iter(20).line_search_fn("strong_wolfe");
    torch::optim::LBFGS optim(params, optim_options);
    auto closure = [&]() -> torch::Tensor {
        optim.zero_grad();

        auto sigma    = torch::exp(raw_sigma);

        auto model_price = DSO::compute_black_scholes_price(s0_tensor, sigma, market_strike_scaled, maturity);
        auto loss = torch::mse_loss(model_price, market_price_scaled);
        
        loss.backward();
        return loss;
    };

    if (verbose) std::cout << "STARTING BLACK-SCHOLES MODEL CALIBRATION\n";
    std::cout << std::fixed << std::setprecision(6);
    try {
        for (int i = 0; i < 15; ++i) {
            auto loss = optim.step(closure);
            
            if (verbose) std::cout << "Iteration " << i << " | MSE: " << loss.item<double>() << "\n";
            if (loss.item<double>() < 1e-9) break; // Convergence threshold
        }
    } catch (const c10::Error& e) {
        std::cerr << "LibTorch Error: " << e.msg() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Error: " << e.what() << std::endl;
    }
    
    BlackScholesModelParameters out;
    out.s0 = s0;
    out.sigma = torch::exp(raw_sigma).item<double>();
    if (verbose) std::cout << "\nCalibrated Parameters:\n"
              << "sigma:    " << out.sigma << "\n";
    return out;
}

class BlackScholesModelImpl final : public StochasticModelImpl {
    public:
        struct Config {
            BlackScholesModelParameters parameters;
            bool use_log_sigma;
        };
        BlackScholesModelImpl(const Config& config)
        : config_(config) {
            auto opt = torch::TensorOptions().dtype(torch::kFloat32);
            s0_tensor_ = register_parameter("s0", torch::tensor({static_cast<float>(config_.parameters.s0)}, opt));
            if (config_.use_log_sigma) {
                log_sigma_tensor_ = register_parameter("log_sigma", torch::tensor({static_cast<float>(std::log(config.parameters.sigma))}, opt));
            } else {
                sigma_tensor_ = register_parameter("sigma", torch::tensor({static_cast<float>(config.parameters.sigma)}, opt));
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
            const EvalContext& ctx
        ) override {
            TORCH_CHECK(init_spec_ != nullptr, "call bind(gridspec) before simulate_batch");
            TORCH_CHECK(ctx.rng.get() != nullptr, "ThreadContext.rng must be set");

            SimulationResult out;
            const auto device = ctx.device;
            const auto dtype = ctx.dtype;
            const int64_t n_steps = dt_.numel();
            const int64_t B = static_cast<int64_t>(batch.n_paths);

            auto* perf = ctx.perf;
            auto opts = torch::TensorOptions().device(device).dtype(dtype);
            torch::Tensor z;
            
            if (device == torch::kCPU) {
                
                std::optional<DSO::ScopedTimer> rng_timer;
                if (perf) rng_timer.emplace(DSO::ScopedTimer(*perf, DSO::Stage::Rng));

                z = torch::empty({B, n_steps}, opts);
                float* z_ptr = z.data_ptr<float>();
                for (int64_t i = 0; i < B; ++i) {
                    const uint64_t path_idx = static_cast<uint64_t>(batch.first_path) + static_cast<uint64_t>(i);
                    ctx.rng->seed_path(path_idx + batch.rng_offset);
                    ctx.rng->fill_normal(z_ptr + i * n_steps, n_steps, 0.0, 1.0);
                }
            } else {
                std::optional<DSO::ScopedTimer> rng_timer;
                if (perf) rng_timer.emplace(DSO::ScopedTimer(*perf, DSO::Stage::Rng));
                z = torch::randn({B, n_steps}, opts);                
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
        torch::Tensor get_sigma_() const { return config_.use_log_sigma ? torch::exp(log_sigma_tensor_) : sigma_tensor_; }

    private:
        const Config& config_;

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
