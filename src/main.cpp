#include <torch/torch.h>
#include "dso.hpp"
#include <iostream>
#include <chrono>

int main() {
    try {
        size_t num_threads = 8;
        size_t n_paths = 1ULL << 14;
        size_t batch_size = 1ULL << 10;
        double maturity = 1.0;
        double strike = 100.0;
        double s0 = 100.0;
        double r = 0.0;
        double sigma = 0.25;
        auto product = DSO::EuropeanCallOption(maturity, strike);
        auto mc_config = DSO::MonteCarloExecutor::Config(
            num_threads,
            batch_size,
            /*seed=*/42
        );
        auto black_scholes_config = DSO::BlackScholesModel::Config(
            mc_config, 
            /*use_log_params=*/true
        );
        auto model = DSO::BlackScholesModel(
            s0,
            r,
            sigma,
            black_scholes_config
        );
        auto objective = DSO::MCPriceObjective(
            7.97f, // Black-Scholes price for sigma = 0.20
            n_paths,
            model,
            product
        );

        model.init(product);
        torch::optim::Adam optim(objective.parameters(), torch::optim::AdamOptions(1e-3));

        auto start = std::chrono::high_resolution_clock::now();

        for (int iter = 0; iter < 1000; ++iter) {
            optim.zero_grad();
            auto loss = objective.forward();
            loss.backward();
            optim.step();
            

            if (iter % 25 == 0) {
                // objective.resample_paths(n_paths);
                std::cout << "iter=" << iter
                        << " loss=" << loss.item<float>() << "\n";
            }
        }

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Time=" << duration.count() << " ms\n";

        auto parameters = objective.parameters();
        auto parameter_names = objective.parameter_names();

        for (int idx = 0; idx < parameters.size(); idx ++) {
            std::cout << parameter_names[idx] << "=" << parameters[idx] << "\n";
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
