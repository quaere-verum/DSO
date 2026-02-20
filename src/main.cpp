#include <torch/torch.h>
#include "dso.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>

int main() {
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);

    try {
        constexpr size_t num_threads = 12;
        constexpr size_t n_paths = 1ULL << 18;
        constexpr size_t batch_size = 4096;
        double maturity = 1.0;
        double strike = 100.0;
        size_t n_steps = static_cast<size_t>(maturity * 365.0);
        double s0 = 100.0;
        double r = 0.0;
        double sigma = 0.20;
        std::vector<double> control_times = DSO::make_time_grid(maturity, maturity / 365.0, true);
        std::cout << "control_times=\n";
        for (auto t : control_times) {
            std::cout << t << "\n";
        }
        // auto product = DSO::AsianCallOption(maturity, strike, n_steps);
        auto product = DSO::EuropeanCallOption(maturity, strike);
        auto mc_config = DSO::MonteCarloExecutor::Config(
            num_threads,
            batch_size,
            /*seed=*/42,
            /*collect_perf*/false
        );
        auto black_scholes_config = DSO::BlackScholesModel::Config(
            /*use_log_params=*/true
        );
        std::vector<double> master_time_grid = DSO::merge_time_grids(control_times, product.time_grid());
        std::cout << "master_time_grid=\n";
        for (auto t : master_time_grid) {
            std::cout << t << "\n";
        }
        DSO::SimulationGridSpec gridpsec;
        gridpsec.include_t0 = product.include_t0() || control_times.front() < 1e-12;
        gridpsec.time_grid = master_time_grid;
        auto feature_extractor = std::make_unique<DSO::OptionFeatureExtractor>();
        auto controller = DSO::LinearHedgeController(
            std::move(feature_extractor),
            DSO::LinearHedgeController::Config(false)
        );
        std::cout << "Controller initiated" << std::endl;
        auto model = DSO::BlackScholesModel(
            s0,
            r,
            sigma,
            black_scholes_config
        );
        std::cout << "Model initiated" << std::endl;
        // auto objective = DSO::MCPriceObjective(
        //     7.97f, // Black-Scholes price for sigma = 0.20
        //     n_paths,
        //     product
        // );
        
        auto objective = DSO::MCHedgeObjective(
            n_paths,
            7.97,
            product, 
            controller,
            control_times
        );
        std::cout << "Objective initiated" << std::endl;
        model.init(gridpsec);
        objective.bind(gridpsec);
        std::cout << "Gridspec bound" << std::endl;
        // auto optim = DSO::LBFGS(torch::optim::LBFGS(controller.parameters(), torch::optim::LBFGSOptions(1.0).line_search_fn("strong_wolfe")));
        auto optim = DSO::Adam(torch::optim::Adam(controller.parameters(), torch::optim::AdamOptions(5e-2)));
        // auto optim = DSO::LBFGS(torch::optim::LBFGS(model.parameters(), torch::optim::LBFGSOptions(1.0).line_search_fn("strong_wolfe")));
        // auto optim = DSO::Adam(torch::optim::Adam(model.parameters(), torch::optim::AdamOptions(1e-3)));
        std::cout << "Optimiser initiated" << std::endl;
        auto trainer = DSO::MonteCarloGradientTrainer(
            DSO::MonteCarloGradientTrainer::Config(
                mc_config,
                n_paths
            ),
            model,
            product,
            objective,
            optim,
            &controller
        );
        std::cout << "Trainer initiated" << std::endl;
        std::cout << "STARTING TRAINING" << std::endl;
        for (int iter = 0; iter < 1000; ++iter) {
            auto start = std::chrono::high_resolution_clock::now();
            torch::Tensor loss = optim.step(trainer);
            
            if (iter % 1 == 0) {
                // objective.resample_paths(n_paths);
                std::cout << "iter=" << iter
                        << " loss=" << loss.item<float>() << "\n";
            }
            auto stop = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
            std::cout << "iter=" << iter << ", time=" << duration.count() << " ms\n";
        }

        std::cout << "FINISHED TRAINING" << std::endl;

        const auto model_parameters = model.parameters();
        const auto model_parameter_names = model.parameter_names();

        for (int idx = 0; idx < model_parameters.size(); idx ++) {
            std::cout << model_parameter_names[idx] << "=\n" << model_parameters[idx] << "\n";
        }

        const auto controller_parameters = controller.parameters();
        const auto controller_parameter_names = controller.parameter_names();
        for (int idx = 0; idx < controller_parameters.size(); idx++) {
            std:: cout << controller_parameter_names[idx] << "=\n" << controller_parameters[idx] << "\n";
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
