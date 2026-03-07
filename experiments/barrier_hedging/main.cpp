#include "runner.hpp"
#include "experiment_context.hpp"
#include <iostream>
#include <torch/torch.h>
#include "expected_payoff.hpp"

int main(int argc, char* argv[]) {
    try {
        ExperimentConfig cfg = parse_args(argc, argv);
        if (cfg.device == torch::kCPU) {
            torch::set_num_interop_threads(1);
            torch::set_num_threads(1);
        }
        std::cout << "EXPERIMENT CONFIG CREATED\n";
        ExperimentContext ctx = create_context(cfg);
        std::cout << "EXPERIMENT CONTEXT CREATED\n";
        // compute_expected_payoff(ctx);
        ctx.config.product_price = 6.67143; // MC valuation
        ExperimentRunner runner(std::move(ctx));
        std::cout << "EXPERIMENT RUNNER CREATED\n";
        runner.run();
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