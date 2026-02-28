#include <torch/torch.h>
#include "dso.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <thread>
#include <tuple>
#include <thread>
#include <cstdlib>


int main() {
    torch::manual_seed(123);
    const size_t cores = std::thread::hardware_concurrency();
    const size_t num_threads = cores;
    std::cout << "cores=" << cores << "\n";
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);

    try {
        constexpr size_t n_paths = 1ULL << 20;
        constexpr size_t batch_size = 1ULL << 14;
        auto mc_config = DSO::MonteCarloExecutor::Config(
            num_threads,
            batch_size,
            /*seed=*/42,
            /*collect_perf*/false
        );
        double maturity = 1.0;
        double strike = 100.0;
        auto product = DSO::EuropeanCallOption(maturity, strike);
        double product_price = 7.97; // Black-Scholes implied vol = 20%
        DSO::HestonModelParameters initial_params;
        initial_params.s0 = 100.0;
        initial_params.v0 = 0.04;
        initial_params.kappa = 2.0;
        initial_params.theta = 0.04;
        initial_params.xi = 0.5;
        initial_params.rho = -0.7;
        auto model_config = DSO::HestonModelImpl::Config(
            initial_params,
            true
        );
        auto model = DSO::HestonModel(model_config);
        
        for (auto& named_param : model->named_parameters()) {
            const auto& name = named_param.key();
            auto& param = named_param.value();
            if (name != "s0") {
                param.requires_grad_(true);
            } else {
                param.requires_grad_(false);
            }
        }
        auto objective = DSO::MCCalibrationObjective(
            product_price,
            product
        );
        DSO::SimulationGridSpec gridspec;
        gridspec.time_grid = product.time_grid();
        gridspec.include_t0 = product.include_t0();
        model->init(gridspec);
        double lr = 1e-1;
        auto optim = DSO::Adam(torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(lr)));
        auto trainer = DSO::MonteCarloGradientTrainer(
            DSO::MonteCarloGradientTrainer::Config(
                mc_config,
                n_paths
            ),
            *model,
            product,
            objective,
            optim
        );
        std::cout << "STARTED CALIBRATION\n";
        for (int iter = 0; iter < 250; ++iter) {
            torch::Tensor loss = optim.step(trainer);
            lr *= 0.985;
            optim.set_lr(lr);
            
            if ((iter + 1) % 10 == 0) {
                std::cout << "iter=" << iter + 1
                        << " loss=" << loss.item<float>() << "\n";
            }
        }

        std::cout << "FINISHED CALIBRATION\n";
        for (const auto& named_param : model->named_parameters()) {
            const auto& name = named_param.key();
            const auto& value = named_param.value();
            std::cout << name << "=\n" << value.detach() << "\n";
        }
        return 0;
    } catch (const c10::Error& e) {
        std::cerr << "Caught c10::Error:\n" << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Caught std::exception:\n" << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n=== UNKNOWN ERROR ===\n";
        return 1;
    }
}
