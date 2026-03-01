#include <torch/torch.h>
#include "dso.hpp"

#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>

namespace {

struct ExperimentConfig {
    size_t n_paths = 1ULL << 20;
    size_t batch_size = 1ULL << 14;
    int64_t seed = 123;
    double hedge_freq = 12.0;         // annual re-hedges
    std::string risk_name = "cvar";
    double risk_alpha = 0.95;         // CVaR level
    double risk_lambda = 1.0;         // entropic or mean-var param
    double learning_rate = 1e-1;
    double lr_decay = 0.98;
    int training_iters = 100;
    torch::Device device = torch::kCPU;

    // Fixed product parameters
    double maturity = 1.0;
    double strike = 100.0;
    double s0 = 100.0;
    double sigma = 0.20;
};

} // namespace

// ============================================================
// Risk Factory
// ============================================================

std::unique_ptr<DSO::RiskMeasure>
make_risk(const ExperimentConfig& cfg) {

    if (cfg.risk_name == "mse")
        return std::make_unique<DSO::MeanSquareRisk>();

    if (cfg.risk_name == "cvar")
        return std::make_unique<DSO::CVaRRisk>(cfg.risk_alpha);

    if (cfg.risk_name == "entropic")
        return std::make_unique<DSO::EntropicRisk>(cfg.risk_lambda);

    if (cfg.risk_name == "meanvar")
        return std::make_unique<DSO::MeanVarianceRisk>(cfg.risk_lambda);

    throw std::invalid_argument("Unknown risk: " + cfg.risk_name);
}

// ============================================================
// Argument Parsing
// ============================================================

ExperimentConfig parse_args(int argc, char* argv[]) {

    ExperimentConfig cfg;

    std::unordered_map<std::string, std::string> args;

    for (int i = 1; i < argc - 1; i += 2) {
        args[argv[i]] = argv[i + 1];
    }

    if (args.count("--paths"))
        cfg.n_paths = std::stoull(args["--paths"]);

    if (args.count("--batch-size"))
        cfg.batch_size = std::stoull(args["--batch-size"]);

    if (args.count("--hedge-freq"))
        cfg.hedge_freq = std::stod(args["--hedge-freq"]);

    if (args.count("--risk"))
        cfg.risk_name = args["--risk"];

    if (args.count("--alpha"))
        cfg.risk_alpha = std::stod(args["--alpha"]);

    if (args.count("--lambda"))
        cfg.risk_lambda = std::stod(args["--lambda"]);

    if (args.count("--lr"))
        cfg.learning_rate = std::stod(args["--lr"]);
    
    if (args.count("--lr-decay"))
        cfg.lr_decay = std::stod(args["--lr-decay"]);

    if (args.count("--iters"))
        cfg.training_iters = std::stoi(args["--iters"]);

    if (args.count("--device"))
        cfg.device = args["--device"] == "cpu" ? torch::kCPU : torch::kCUDA;

    if (cfg.hedge_freq <= 0.0)
        throw std::invalid_argument("hedge-freq must be positive");

    if (cfg.n_paths == 0)
        throw std::invalid_argument("paths must be > 0");
    
    if (cfg.batch_size == 0)
        throw std::invalid_argument("batch_size must be > 0");
    
    if (cfg.lr_decay <= 0.0 || cfg.lr_decay > 1.0)
        throw std::invalid_argument("lr_decay must be in (0, 1]");

    if (cfg.learning_rate <= 0.0)
        throw std::invalid_argument("learning_rate must be > 0.0");

    return cfg;
}

// ============================================================
// Main
// ============================================================

int main(int argc, char* argv[]) {

    torch::manual_seed(123);

    size_t cores = std::thread::hardware_concurrency();
    std::cout << "CPU cores = " << cores << "\n";

    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);

    try {

        auto cfg = parse_args(argc, argv);

        std::cout << "\n===== Experiment Configuration =====\n";
        std::cout << "Device        : " << cfg.device << "\n";
        std::cout << "Paths         : " << cfg.n_paths << "\n";
        std::cout << "Hedge freq    : " << cfg.hedge_freq << "\n";
        std::cout << "Risk          : " << cfg.risk_name << "\n";
        std::cout << "Learning rate : " << cfg.learning_rate << "\n";
        std::cout << "Iterations    : " << cfg.training_iters << "\n\n";

        // ----------------------------------------------------
        // Calibrate Model Parameters
        // ----------------------------------------------------
        auto device = cfg.device;
        auto opt = torch::TensorOptions().dtype(torch::kFloat32);
        torch::Tensor market_strikes = torch::tensor({90.0, 95.0, 100.0, 105.0, 110.0}, opt);
        torch::Tensor market_prices = torch::tensor({14.93, 11.27, 7.97, 5.51, 3.56}, opt);

        torch::Tensor hedged_strike_price;
        for (size_t i = 0; i < market_prices.size(0); ++i) {
            auto strike = market_strikes[i].item<double>();
            if (std::fabs(strike - cfg.strike) < 1e-6) {
                hedged_strike_price = market_prices[i].clone();
                break;
            }
        }
        TORCH_CHECK(hedged_strike_price.defined(), "Strike not found in market strikes.")
        double product_price = hedged_strike_price.item<double>();

        auto bs_params = DSO::calibrate_black_scholes_model_parameters(
            hedged_strike_price.item<double>(),
            cfg.strike,
            100.0,
            cfg.maturity,
            true
        );

        auto bs_config = DSO::BlackScholesModelImpl::Config(
            bs_params,
            false
        );

        // ----------------------------------------------------
        // Model + Product
        // ----------------------------------------------------

        auto product = DSO::EuropeanCallOption(cfg.maturity, cfg.strike);

        auto black_scholes_model = DSO::BlackScholesModel(bs_config);
        black_scholes_model->to(device);
        black_scholes_model->eval();
        for (auto& p : black_scholes_model->parameters()) p.requires_grad_(false);
        std::cout << "MODEL CREATED" << std::endl;

        // ----------------------------------------------------
        // Control grid
        // ----------------------------------------------------

        const double dt = cfg.maturity / cfg.hedge_freq;
        auto control_times = DSO::make_time_grid(cfg.maturity, dt, true);

        DSO::ControlIntervals intervals;
        intervals.start_times.assign(
            control_times.begin(),
            control_times.end() - 1
        );
        intervals.end_times.assign(
            control_times.begin() + 1,
            control_times.end()
        );

        auto master_grid = DSO::merge_time_grids(control_times, product.time_grid());

        DSO::SimulationGridSpec gridspec;
        gridspec.include_t0 = product.include_t0() || control_times.front() < 1e-12;
        gridspec.time_grid = master_grid;
        black_scholes_model->init(gridspec);

        // ----------------------------------------------------
        // Controller
        // ----------------------------------------------------

        auto feature_extractor = DSO::OptionFeatureExtractor(product);
        feature_extractor->to(device);
        
        auto black_scholes_controller = DSO::MlpController(DSO::MlpControllerImpl::Config(feature_extractor->feature_dim(), {}));
        black_scholes_controller->to(device);
        for (auto& p : black_scholes_controller->parameters()) p.requires_grad_(true);
        std::cout << "CONTROLLER MOVED TO GPU" << std::endl;

        // ----------------------------------------------------
        // Hedging Engine
        // ----------------------------------------------------

        auto hedging_engine = DSO::HedgingEngine(product_price, intervals);
        hedging_engine.bind(gridspec);

        // ----------------------------------------------------
        // Risk
        // ----------------------------------------------------

        auto risk = make_risk(cfg);

        // ----------------------------------------------------
        // Objective + Trainer
        // ----------------------------------------------------

        auto black_scholes_objective = DSO::MCHedgeObjective(
            product,
            *black_scholes_controller,
            hedging_engine,
            *risk,
            *feature_extractor
        );

        auto black_scholes_optim = DSO::Adam(
            torch::optim::Adam(
                black_scholes_controller->parameters(),
                torch::optim::AdamOptions(cfg.learning_rate)
            )
        );


        auto mc_config = DSO::MonteCarloExecutor::Config(cores, cfg.batch_size, cfg.seed, false, device);
        auto black_scholes_trainer =
            DSO::MonteCarloGradientTrainer(
                {mc_config, cfg.n_paths, device},
                *black_scholes_model,
                product,
                black_scholes_objective,
                black_scholes_optim
            );
        // ----------------------------------------------------
        // Black-Scholes Training Loop
        // ----------------------------------------------------

        std::cout << "\nSTART BLACK-SCHOLES TRAINING" << std::endl;;

        double lr = cfg.learning_rate;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < cfg.training_iters; ++iter) {

            auto loss = black_scholes_optim.step(black_scholes_trainer);

            lr *= cfg.lr_decay;
            black_scholes_optim.set_lr(lr);

            if ((iter + 1) % 10 == 0) {
                std::cout << "iter=" << iter + 1
                          << " loss="
                          << loss.item<double>()
                          << "\n";
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "TRAINING FINISHED" << std::endl;
        std::cout << "Duration="  
        << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) * 1e-3 << "s\n";

        // ----------------------------------------------------
        // Print trained params
        // ----------------------------------------------------

        for (const auto& p : black_scholes_controller->named_parameters()) {
            std::cout << p.key() << "=\n"
                      << p.value() << "\n";
        }
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}