#include <torch/torch.h>
#include "dso.hpp"

#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>
#include "feature_extractors.hpp"
#include "config.hpp"
#include "training.hpp"
#include "evaluation.hpp"
#include "linreg_benchmark.hpp"


int main(int argc, char* argv[]) {

    torch::manual_seed(123);

    size_t cores = std::thread::hardware_concurrency();
    std::cout << "CPU cores = " << cores << "\n";

    try {

        auto cfg = parse_args(argc, argv);

        std::cout << "\n===== Experiment Configuration =====\n";
        std::cout << "Device        : " << cfg.device << "\n";
        std::cout << "Paths         : " << cfg.n_paths << "\n";
        std::cout << "Hedge freq    : " << cfg.hedge_freq << "\n";
        std::cout << "Risk          : " << cfg.risk_name << "\n";
        std::cout << "Learning rate : " << cfg.learning_rate << "\n";
        std::cout << "Iterations    : " << cfg.training_iters << "\n\n";

        if (cfg.device == torch::kCPU) {
            torch::set_num_threads(1);
            torch::set_num_interop_threads(1);
        }

        // ----------------------------------------------------
        // Model + Product
        // ----------------------------------------------------

        auto product = DSO::EuropeanCallOption(cfg.maturity, cfg.strike);

        auto model = DSO::HestonModel(
            DSO::HestonModelImpl::Config(
                {
                    /*s0*/100.0,
                    /*v0*/0.054094,
                    /*kappa*/1.514632,
                    /*theta*/0.053657,
                    /*xi*/0.513620,
                    /*rho*/-0.861537
                }, 
                false
            )
        );
        model->to(cfg.device);

        const double dt = cfg.maturity / cfg.hedge_freq;
        auto control_times = DSO::make_time_grid(cfg.maturity, dt, true);

        auto feature_extractor = RecurrentOptionFeatureExtractor(product);
        feature_extractor->to(cfg.device);
        auto controller = DSO::MlpController(DSO::MlpControllerImpl::Config(feature_extractor->feature_dim(), cfg.hidden_sizes)).ptr();
        controller->to(cfg.device);
    
        train_hedge_parameters(
            product,
            *model,
            *feature_extractor,
            *controller,
            control_times,
            cfg
        );

        std::cout << "FEATURE PARAMETERS\n";
        for (const auto& p : feature_extractor->named_parameters()) {
            std::cout << p.key() << "=\n"
                      << p.value() << "\n";
        }
        std::cout << "CONTROLLER PARAMETERS\n";
        for (const auto& p : controller->named_parameters()) {
            std::cout << p.key() << "=\n"
                      << p.value() << "\n";
        }

        auto loss = eval_hedge_parameters(
            product,
            *model,
            *feature_extractor,
            *controller,
            control_times,
            cfg
        );
        auto linreg_feature_extractor = DSO::OptionFeatureExtractor(product);
        linreg_feature_extractor->to(cfg.device);
        auto linreg_controller = linear_regression_benchmark(
            product,
            *model,
            *linreg_feature_extractor,
            control_times,
            cfg
        );
        std::cout << "LINREG CONTROLLER PARAMETERS\n";
        for (const auto& p : linreg_controller->named_parameters()) {
            std::cout << p.key() << "=\n"
                      << p.value() << "\n";
        }

        auto linreg_loss = eval_hedge_parameters(
            product,
            *model,
            *linreg_feature_extractor,
            *linreg_controller,
            control_times,
            cfg
        );

        std::cout << "DSO    loss = " << loss << "\n";
        std::cout << "Linreg loss = " << linreg_loss << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
