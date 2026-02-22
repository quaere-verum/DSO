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

void valuation(
    size_t n_paths,
    DSO::MonteCarloExecutor::Config mc_config,
    DSO::Option& product
) {
    double s0 = 100.0;
    double r = 0.0;
    double sigma = 0.20;
    auto model = DSO::BlackScholesModel(s0, r, sigma);
    DSO::SimulationGridSpec gridspec;
    gridspec.include_t0 = product.include_t0();
    for (auto t : product.time_grid()) {
        gridspec.time_grid.push_back(t);
    }
    model.init(gridspec);
    model.set_mode(DSO::ModelEvalMode::VALUATION);

    std::vector<std::tuple<std::string, std::string>> second_order_derivatives = {
        {"s0", "s0"}, 
        {"s0", "sigma"}
    };
    auto valuator = DSO::MonteCarloValuation(
        DSO::MonteCarloValuation::Config(mc_config, n_paths, second_order_derivatives),
        model
    );
    auto start = std::chrono::high_resolution_clock::now();
    auto value = valuator.value(product);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    std::cout << "Valuation duration=" << duration << "ms\n";
    auto names = model.parameter_names();
    std::cout << "value=" << value.value << "\n";
    for (size_t i = 0; i < names.size(); ++i) {
        std::cout << names[i] << "=" << value.gradient[i] << "\n";
    }
    for (size_t i = 0; i < second_order_derivatives.size(); ++i) {
        std::cout 
        << std::get<0>(second_order_derivatives[i])
        << ", "
        << std::get<1>(second_order_derivatives[i]) 
        << "=" << value.second_order_derivatives[i] << "\n";
    }
}

void hedging(
    size_t n_paths,
    DSO::MonteCarloExecutor::Config mc_config,
    DSO::Option& product
) {
    double s0 = 100.0;
    double r = 0.0;
    double sigma = 0.20;
    auto model = DSO::BlackScholesModel(s0, r, sigma);
    const double maturity = product.maturity();
    std::vector<double> control_times = DSO::make_time_grid(maturity, maturity / 365.0, true);
    DSO::ControlIntervals control_intervals;
    control_intervals.start_times = std::vector<double>(
        control_times.begin(),
        control_times.end() - 1
    );
    control_intervals.end_times = std::vector<double>(
        control_times.begin() + 1,
        control_times.end()
    );
    std::vector<double> master_time_grid = DSO::merge_time_grids(control_times, product.time_grid());
    DSO::SimulationGridSpec gridspec;
    gridspec.include_t0 = product.include_t0() || control_times.front() < 1e-12;
    gridspec.time_grid = master_time_grid;
    auto feature_extractor = std::make_unique<DSO::OptionFeatureExtractor>(product);
    auto controller = DSO::LinearHedgeController(
        std::move(feature_extractor),
        DSO::LinearHedgeController::Config(false)
    );
    std::cout << "Controller initiated" << std::endl;
    auto objective = DSO::MCHedgeObjective(
        n_paths,
        7.97,
        product, 
        controller,
        control_intervals
    );
    std::cout << "Objective initiated" << std::endl;
    model.init(gridspec);
    objective.bind(gridspec);
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
}

void calibration(
    size_t n_paths,
    DSO::MonteCarloExecutor::Config mc_config,
    DSO::Option& product
) {
    double s0 = 100.0;
    double r = 0.0;
    double sigma = 0.20;
    auto model = DSO::BlackScholesModel(s0, r, sigma);
    auto objective = DSO::MCPriceObjective(
        7.97f, // Black-Scholes price for sigma = 0.20 and K=100
        n_paths,
        product
    );
}

int main() {
    const size_t cores = std::thread::hardware_concurrency();
    const size_t num_threads = cores;
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);

    try {
        constexpr size_t n_paths = 1ULL << 18;
        constexpr size_t batch_size = 1ULL << 13;
        double maturity = 1.0;
        double strike = 100.0;
        size_t n_steps = static_cast<size_t>(maturity * 365.0);

        // auto product = DSO::AsianCallOption(maturity, strike, n_steps);
        auto product = DSO::EuropeanCallOption(maturity, strike);
        auto mc_config = DSO::MonteCarloExecutor::Config(
            num_threads,
            batch_size,
            /*seed=*/42,
            /*collect_perf*/false
        );

        valuation(n_paths, mc_config, product);
        // hedging(n_paths, mc_config, product);
        // calibration(n_paths, mc_config, product);

        

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
