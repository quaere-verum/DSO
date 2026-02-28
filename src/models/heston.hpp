#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <complex>

#include "models/stochastic_model.hpp"
#include "core/threading.hpp"
#include "products/product.hpp"
#include "simulation/monte_carlo.hpp"

namespace DSO {

struct HestonModelParameters {
    double s0;
    double v0;
    double kappa;
    double theta;
    double xi;
    double rho;
};

// Stable Heston Characteristic Function (Albrecher et al. version)
torch::Tensor heston_cf(
    torch::Tensor u, 
    torch::Tensor T, 
    torch::Tensor s0, 
    torch::Tensor v0, 
    torch::Tensor kappa, 
    torch::Tensor theta, 
    torch::Tensor xi, 
    torch::Tensor rho
) {
    auto i = torch::complex(torch::tensor(0.0), torch::tensor(1.0)).to(u.device());
    
    // Albrecher version parameters
    auto d = torch::sqrt(torch::pow(rho * xi * i * u - kappa, 2) + 
                        torch::pow(xi, 2) * (i * u + torch::pow(u, 2)));
    d = torch::where(torch::real(d) < 0, -d, d);
    auto g = (kappa - rho * xi * i * u - d) / (kappa - rho * xi * i * u + d);
    
    auto exp_dt = torch::exp(-d * T);
    auto G = (1.0 - g * exp_dt) / (1.0 - g);
    
    auto C = (kappa * theta / torch::pow(xi, 2)) * ((kappa - rho * xi * i * u - d) * T - 2.0 * torch::log(G));
    
    auto D = ((kappa - rho * xi * i * u - d) / torch::pow(xi, 2)) * ((1.0 - exp_dt) / (1.0 - g * exp_dt));
    
    // characteristic function: exp(C + D*v0 + i*u*log(s0))
    return torch::exp(C + D * v0 + i * u * torch::log(s0));
}

torch::Tensor compute_heston_prices(
    torch::Tensor s0, 
    torch::Tensor v0, 
    torch::Tensor kappa, 
    torch::Tensor theta, 
    torch::Tensor xi, 
    torch::Tensor rho,
    double T_val, 
    torch::Tensor market_strikes,
    double alpha = 1.5,
    int num_points = 1000,
    double kMaxU = 100.0  // Upper limit for integration
) {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(s0.device());
    auto T = torch::tensor(T_val, options);
    auto i = torch::complex(torch::tensor(0.0, options), torch::tensor(1.0, options));
    
    // 1. Create a fine grid for the integration variable 'u'
    // We only integrate from 0 to kMaxU because the integrand decays rapidly
    auto u_grid = torch::linspace(0.0, kMaxU, num_points, options);
    
    std::vector<torch::Tensor> prices;

    for (int64_t j = 0; j < market_strikes.size(0); ++j) {
        auto K = market_strikes[j];
        auto logK = torch::log(K);

        // 2. Compute the dampened integrand for this specific strike
        // phi(u - (alpha + 1)i)
        auto u_shifted = u_grid - (alpha + 1.0) * i;
        auto phi = heston_cf(u_shifted, T, s0, v0, kappa, theta, xi, rho);
        
        // The Carr-Madan formula for the dampened integrand
        auto denominator = (alpha + i * u_grid) * (alpha + 1.0 + i * u_grid);
        auto exponential_term = torch::exp(-i * u_grid * logK);
        
        auto complex_integrand = (exponential_term * phi / denominator);
        auto real_integrand = torch::real(complex_integrand);

        // 3. Integrate using the Trapezoidal rule
        // Integral results in: (exp(-alpha * logK) / PI) * integral(integrand * du)
        auto integral_val = torch::trapezoid(real_integrand, u_grid);
        auto price = (torch::exp(-alpha * logK) / M_PI) * integral_val;
        
        prices.push_back(price);
    }

    return torch::stack(prices);
}

torch::Tensor compute_heston_prices_fft(
    torch::Tensor s0, 
    torch::Tensor v0, 
    torch::Tensor kappa, 
    torch::Tensor theta, 
    torch::Tensor xi, 
    torch::Tensor rho,
    double T_val, 
    torch::Tensor market_strikes,
    int N = 4096,      // Must be power of 2
    double B = 500.0,  // Upper bound for integration
    double alpha = 1.5 // Damping factor

) {
    auto device = s0.device();
    auto T = torch::tensor(T_val).to(device);

    // 1. Setup Grid
    double du = B / N;
    double dk = (2.0 * M_PI) / B;
    double k0 = - (dk * N) / 2.0; // Log-strike grid start

    auto u_grid = torch::linspace(0, (N - 1) * du, N).to(device);
    auto k_grid = torch::linspace(k0, k0 + (N - 1) * dk, N).to(device);

    // 2. Compute dampened integrand: 
    // psi = exp(-rT) * phi(u - (alpha+1)i) / (alpha + iu)(alpha + 1 + iu)
    auto i = torch::complex(torch::tensor(0.0), torch::tensor(1.0)).to(device);
    auto u_shifted = u_grid - (alpha + 1.0) * i;
    
    auto phi = heston_cf(u_shifted, T, s0, v0, kappa, theta, xi, rho);
    
    auto denominator = (alpha + i * u_grid) * (alpha + 1.0 + i * u_grid);
    auto integrand = (phi / denominator) * torch::exp(-i * u_grid * k0);

    // Simpson's rule weights
    auto weights = torch::ones({N}).to(device);
    weights.index_put_({torch::indexing::Slice(1, -1, 2)}, 4.0 / 3.0);
    weights.index_put_({torch::indexing::Slice(2, -1, 2)}, 2.0 / 3.0);
    weights.index_put_({0}, 1.0 / 3.0);
    weights.index_put_({N - 1}, 1.0 / 3.0);
    integrand *= weights * du;

    // 3. FFT
    auto fft_res = torch::fft::fft(integrand);
    
    // Extract call prices: C(k) = exp(-alpha*k) / pi * Re(fft_res)
    auto call_prices_grid = (torch::exp(-alpha * k_grid) / M_PI) * torch::real(fft_res);

    // 4. Interpolate to market strikes
    // Convert market strikes to log-space
    auto log_market_strikes = torch::log(market_strikes);
    
    // Using a basic linear interpolation for LibTorch
    auto idx = ((log_market_strikes - k0) / dk).to(torch::kLong);
    auto weight = (log_market_strikes - (k0 + idx.to(torch::kFloat) * dk)) / dk;
    
    auto p0 = call_prices_grid.index({idx});
    auto p1 = call_prices_grid.index({idx + 1});
    
    return torch::lerp(p0, p1, weight);
}

HestonModelParameters calibrate_heston_model_parameters(
    torch::Tensor market_prices, 
    torch::Tensor market_strikes,
    double s0,
    double maturity,
    bool verbose = false
) {
    auto market_prices_scaled = market_prices / s0;
    auto market_strikes_scaled = market_strikes / s0;

    auto s0_tensor = torch::tensor(1.0, market_prices.options());
    auto raw_v0    = torch::log(torch::tensor(0.04, market_prices.options())).set_requires_grad(true);
    auto raw_kappa = torch::log(torch::tensor(2.0, market_prices.options())).set_requires_grad(true);
    auto raw_theta = torch::log(torch::tensor(0.04, market_prices.options())).set_requires_grad(true);
    auto raw_xi    = torch::log(torch::tensor(0.3, market_prices.options())).set_requires_grad(true);
    auto raw_rho   = torch::atanh(torch::tensor(-0.7, market_prices.options())).set_requires_grad(true);

    std::vector<torch::Tensor> params = {raw_v0, raw_kappa, raw_theta, raw_xi, raw_rho};
    auto optim_options = torch::optim::LBFGSOptions().lr(1.0).max_iter(20).line_search_fn("strong_wolfe");
    torch::optim::LBFGS optim(params, optim_options);
    auto closure = [&]() -> torch::Tensor {
        optim.zero_grad();

        auto v0    = torch::exp(raw_v0);
        auto kappa = torch::exp(raw_kappa);
        auto theta = torch::exp(raw_theta);
        auto xi    = torch::exp(raw_xi);
        auto rho   = torch::tanh(raw_rho);

        // Use either compute_heston_prices or compute_heston_prices_fft
        auto model_prices = DSO::compute_heston_prices(s0_tensor, v0, kappa, theta, xi, rho, maturity, market_strikes_scaled);
        auto loss = torch::mse_loss(model_prices, market_prices_scaled);
        
        loss.backward();
        return loss;
    };

    if (verbose) std::cout << "STARTING HESTON MODEL CALIBRATION\n";
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
    
    HestonModelParameters out;
    out.s0 = s0;
    out.v0 = torch::exp(raw_v0).item<double>();
    out.kappa = torch::exp(raw_kappa).item<double>();
    out.theta = torch::exp(raw_theta).item<double>();
    out.xi = torch::exp(raw_xi).item<double>();
    out.rho = torch::tanh(raw_rho).item<double>();
    if (verbose) std::cout << "\nCalibrated Parameters:\n"
              << "v0:    " << out.v0 << "\n"
              << "kappa: " << out.kappa << "\n"
              << "theta: " << out.theta << "\n"
              << "xi:    " << out.xi << "\n"
              << "rho:   " << out.rho << "\n";
    return out;
}

class HestonModelImpl final : public StochasticModelImpl {
    public:
        struct Config {
            HestonModelParameters parameters;
            bool use_logit_params;
        };
        HestonModelImpl(
            const Config& config
        )
        : config_(config) {
            auto opt = torch::TensorOptions().dtype(torch::kFloat32);

            s0_tensor_ = register_parameter("s0", torch::tensor({(float)config_.parameters.s0}, opt));
            v0_tensor_ = register_parameter("v0", torch::tensor({(float)config_.parameters.v0}, opt));

            if (config_.use_logit_params) {
                log_kappa_ = register_parameter("log_kappa", torch::tensor({(float)std::log(config_.parameters.kappa)}, opt));
                log_theta_ = register_parameter("log_theta", torch::tensor({(float)std::log(config_.parameters.theta)}, opt));
                log_xi_    = register_parameter("log_xi", torch::tensor({(float)std::log(config_.parameters.xi)}, opt));
                logit_rho_ = register_parameter("logit_rho", torch::tensor({(float)std::atanh(config_.parameters.rho)}, opt));
            } else {
                kappa_ = register_parameter("kappa", torch::tensor({(float)config_.parameters.kappa}, opt));
                theta_ = register_parameter("theta", torch::tensor({(float)config_.parameters.theta}, opt));
                xi_    = register_parameter("xi", torch::tensor({(float)config_.parameters.xi}, opt));
                rho_   = register_parameter("rho", torch::tensor({(float)config_.parameters.rho}, opt));
            }
        }

        void init(const SimulationGridSpec& spec) override {
            const auto& time_grid = spec.time_grid;
            const int64_t n_times = static_cast<int64_t>(time_grid.size());
            const int64_t n_steps = n_times - 1;

            std::vector<float> dt_host(n_steps);
            for (int64_t j = 0; j < n_steps; ++j) {
                double dt = time_grid[j+1] - time_grid[j];
                TORCH_CHECK(dt > 0, "dt must be > 0");
                dt_host[j] = static_cast<float>(dt);
            }

            dt_ = torch::from_blob(dt_host.data(), {n_steps},
                torch::TensorOptions().dtype(torch::kFloat32)).clone();
            sqrt_dt_ = torch::sqrt(dt_);
            init_spec_ = &spec;
        }

        SimulationResult simulate_batch(
            const BatchSpec& batch,
            const EvalContext& ctx
        ) override {
            TORCH_CHECK(init_spec_ != nullptr, "call init before simulate");
            TORCH_CHECK(ctx.rng.get() != nullptr, "rng must be set");

            SimulationResult out;
            const int64_t B = batch.n_paths;
            const int64_t n_steps = dt_.numel();

            auto opt = torch::TensorOptions().dtype(torch::kFloat32);
            auto z1 = torch::empty({B, n_steps}, opt);
            auto z2 = torch::empty({B, n_steps}, opt);

            float* z1_ptr = z1.data_ptr<float>();
            float* z2_ptr = z2.data_ptr<float>();

            // --- RNG ---
            for (int64_t i = 0; i < B; ++i) {
                uint64_t path_idx = (uint64_t)batch.first_path + (uint64_t)i;
                ctx.rng->seed_path(path_idx + batch.rng_offset);
                ctx.rng->fill_normal(z1_ptr + i*n_steps, n_steps, 0.0, 1.0);
                ctx.rng->fill_normal(z2_ptr + i*n_steps, n_steps, 0.0, 1.0);
            }

            // --- Correlate Brownian motions ---
            auto rho = get_rho_();
            auto z_v = z1;
            auto z_s = rho * z1 + torch::sqrt(1 - rho * rho) * z2;

            auto kappa = get_kappa_();
            auto theta = get_theta_();
            auto xi    = get_xi_();

            auto logS = torch::empty({B, n_steps+1}, opt);
            auto var  = torch::empty({B, n_steps+1}, opt);

            logS.index_put_({torch::indexing::Slice(), 0}, s0_tensor_.log());
            var.index_put_({torch::indexing::Slice(), 0}, v0_tensor_);

            for (int64_t t = 0; t < n_steps; ++t) {

                auto v_prev = var.index({torch::indexing::Slice(), t});
                auto v_pos  = torch::clamp_min(v_prev, 0.0);

                // variance update (full truncation)
                auto dv =
                    kappa * (theta - v_pos) * dt_[t]
                    + xi * torch::sqrt(v_pos * dt_[t]) * z_v.index({torch::indexing::Slice(), t});

                var.index_put_({torch::indexing::Slice(), t + 1}, torch::clamp_min(v_prev + dv, 0.0));

                // log-forward update
                auto dlogS = -0.5 * v_pos * dt_[t] + torch::sqrt(v_pos * dt_[t]) * z_s.index({torch::indexing::Slice(), t});
                logS.index_put_({torch::indexing::Slice(), t + 1}, logS.index({torch::indexing::Slice(), t}) + dlogS);
            }

            auto spot = torch::exp(logS);

            if (!init_spec_->include_t0) {
                spot = spot.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)});
                var = var.index({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)});
            }

            out.spot = spot;
            out.variance = var;
            return out;
        }

        const std::vector<DSO::FactorType>& factors() const override {
            return factors_;
        }

    private:
        torch::Tensor get_kappa_() const {
            return config_.use_logit_params ? torch::exp(log_kappa_) : kappa_;
        }
        torch::Tensor get_theta_() const {
            return config_.use_logit_params ? torch::exp(log_theta_) : theta_;
        }
        torch::Tensor get_xi_() const {
            return config_.use_logit_params ? torch::exp(log_xi_) : xi_;
        }
        torch::Tensor get_rho_() const {
            return config_.use_logit_params ? torch::tanh(logit_rho_) : rho_;
        }

    private:
        const Config& config_;

        torch::Tensor dt_;
        torch::Tensor sqrt_dt_;
        const SimulationGridSpec* init_spec_ = nullptr;

        torch::Tensor s0_tensor_;
        torch::Tensor v0_tensor_;

        torch::Tensor log_kappa_, log_theta_, log_xi_, logit_rho_;
        torch::Tensor kappa_, theta_, xi_, rho_;

        std::vector<DSO::FactorType> factors_ = {DSO::FactorType::Spot};
};

TORCH_MODULE(HestonModel);

} // namespace DSO