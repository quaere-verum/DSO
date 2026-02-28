#include <torch/torch.h>
#include "dso.hpp"

#include <iostream>
#include <thread>
#include <vector>
#include <memory>
#include <chrono>
#include <algorithm>


class RecurrentOptionFeatureExtractorImpl final : public DSO::FeatureExtractorImpl {

public:
    RecurrentOptionFeatureExtractorImpl(const DSO::Option& option)
        : option_(option)
    {
        auto opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCPU);

        strike_inv_ = 1.0 / option_.strike();
        maturity_   = torch::tensor(option_.maturity(), opts);

        alpha_raw_ = register_parameter("alpha_raw", torch::zeros({1}, opts));
        beta_ = register_parameter("beta", torch::zeros({1}, opts));
        bias_ = register_parameter("bias", torch::zeros({1}, opts));
    }

    DSO::FeatureExtractorResult forward(const DSO::SimulationState& state) const override {

        const auto& S      = state.spot;           // (batch,)
        const auto& S_prev = state.spot_previous;  // (batch,)
        const auto& h_prev = state.hidden_state;   // (batch,1)
        const auto& t      = state.t;
        const auto& t_next = state.t_next;

        const int64_t n_paths = S.size(0);

        auto logS      = torch::log(S);
        auto logS_prev = torch::log(S_prev);
        auto dlogS     = logS - logS_prev;

        auto dt = (t_next - t);
        auto sq_ret = (dlogS * dlogS) / dt;
        sq_ret = sq_ret.unsqueeze(1);

        // ---- Stable EWMA update ----
        auto alpha = torch::sigmoid(alpha_raw_);

        torch::Tensor next_hidden =
            alpha * h_prev
            + beta_ * sq_ret
            + bias_;

        // ---- Features ----
        auto log_moneyness = torch::log(S * strike_inv_).unsqueeze(1);
        auto tau = (maturity_ - t).expand({n_paths, 1});

        torch::Tensor features = torch::cat({log_moneyness, tau, next_hidden}, 1);

        DSO::FeatureExtractorResult out;
        out.features     = features;
        out.hidden_state = next_hidden;

        return out;
    }

    const size_t feature_dim() const override { return 3; }

    const std::optional<size_t> hidden_state_dim() const override { return 1; }

private:
    const DSO::Option& option_;

    double strike_inv_;
    torch::Tensor maturity_;

    torch::Tensor alpha_raw_;  // decay (sigmoid)
    torch::Tensor beta_;       // return weight
    torch::Tensor bias_;       // offset
};
TORCH_MODULE(RecurrentOptionFeatureExtractor);

class HestonOptionFeatureExtractorImpl final : public DSO::FeatureExtractorImpl {

public:
    HestonOptionFeatureExtractorImpl(const DSO::Option& option)
        : option_(option)
    {
        auto opts = torch::TensorOptions()
                        .dtype(torch::kFloat32)
                        .device(torch::kCPU);

        strike_inv_ = 1.0 / option_.strike();
        maturity_   = torch::tensor(option_.maturity(), opts);

        alpha_raw_ = register_parameter("alpha_raw", torch::zeros({1}, opts));
        beta_ = register_parameter("beta", torch::zeros({1}, opts));
        bias_ = register_parameter("bias", torch::zeros({1}, opts));
    }

    DSO::FeatureExtractorResult forward(const DSO::SimulationState& state) const override {

        const auto& S      = state.spot;           // (batch,)
        const auto& v      = state.variance;
        const auto& t      = state.t;
        const auto& t_next = state.t_next;

        const int64_t n_paths = S.size(0);

        // ---- Features ----
        auto log_moneyness = torch::log(S * strike_inv_).unsqueeze(1);
        auto tau = (maturity_ - t).expand({n_paths, 1});

        torch::Tensor features = torch::cat({log_moneyness, tau, v.unsqueeze(1)}, 1);

        DSO::FeatureExtractorResult out;
        out.features = features;
        return out;
    }

    const size_t feature_dim() const override { return 3; }

    const std::optional<size_t> hidden_state_dim() const override { return std::nullopt; }

private:
    const DSO::Option& option_;

    double strike_inv_;
    torch::Tensor maturity_;

    torch::Tensor alpha_raw_;  // decay (sigmoid)
    torch::Tensor beta_;       // return weight
    torch::Tensor bias_;       // offset
};
TORCH_MODULE(HestonOptionFeatureExtractor);

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
    double lr_decay = 0.985;
    double min_lr = 1e-3;
    int training_iters = 100;

    // Fixed product parameters
    double maturity = 1.0;
    double strike = 100.0;
    double product_price = 7.97;
};

} // namespace


// ============================================================
// Benchmark utility
// ============================================================

void benchmark_against_linear(
    const DSO::SimulationResult& simulated,
    const DSO::Option& product,
    const DSO::ControllerImpl& trained_controller,
    const std::vector<double>& control_times,
    const DSO::ControlIntervals& control_intervals,
    const DSO::SimulationGridSpec& gridspec,
    const DSO::RiskMeasure& risk,
    const DSO::FeatureExtractorImpl& trained_feature_extractor,
    const ExperimentConfig& cfg
) {
    auto opt = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(torch::kCPU);

    auto payoff = product.compute_payoff(simulated);
    auto premium = torch::full({(int64_t)cfg.n_paths}, cfg.product_price, opt);

    auto control_indices = DSO::bind_to_grid(control_intervals, gridspec.time_grid);

    // --------------------------------------------------------
    // Compute linear regression features
    // --------------------------------------------------------

    torch::Tensor A1 = torch::zeros({(int64_t)cfg.n_paths}, opt);
    torch::Tensor A2 = torch::zeros({(int64_t)cfg.n_paths}, opt);
    torch::Tensor A3 = torch::zeros({(int64_t)cfg.n_paths}, opt);
    torch::Tensor A4 = torch::zeros({(int64_t)cfg.n_paths}, opt);

    for (size_t k = 0; k < control_intervals.n_intervals(); ++k) {

        int64_t t0 = control_indices.start_idx[k];
        int64_t t1 = control_indices.end_idx[k];

        auto S0 = simulated.spot.select(1, t0);
        auto S1 = simulated.spot.select(1, t1);
        auto v  = simulated.variance.select(1, t0);

        auto dS = S1 - S0;
        auto x = torch::log(S0 / product.strike());
        double tau = product.maturity() - control_times[k];

        A1 += x * dS;
        A2 += torch::full_like(dS, tau) * dS;
        A3 += dS;
        A4 += v * dS;
    }

    auto X = torch::stack({A1, A2, A3, A4}, 1);
    auto y = payoff.to(torch::kFloat32) - premium;

    // --------------------------------------------------------
    // Solve linear regression
    // --------------------------------------------------------

    auto XtX = torch::matmul(X.t(), X);
    auto Xty = torch::matmul(X.t(), y.unsqueeze(1));
    auto w = torch::linalg_solve(XtX, Xty).squeeze();

    double w1 = w[0].item<double>();
    double w2 = w[1].item<double>();
    double b  = w[2].item<double>();
    double w3 = w[3].item<double>();

    std::cout << "\nLinear hedge coefficients:\n"
              << "w1=" << w1 << "\n"
              << "w2=" << w2 << "\n"
              << "w3=" << w3 << "\n"
              << "b =" << b << "\n";

    // --------------------------------------------------------
    // Construct linear controller with fitted weights
    // --------------------------------------------------------

    auto feature_extractor = HestonOptionFeatureExtractor(product);
    auto linear_controller = DSO::LinearController(feature_extractor->feature_dim());
    for (auto& param : linear_controller->named_parameters()) {
        auto& name = param.key();
        auto& tensor = param.value();
        if (name == "b") {
            tensor.data().copy_(torch::tensor({(float)b}, opt));
        } else {
            tensor.data().copy_(torch::tensor({(float)w1, (float)w2, (float)w3}, opt));
        }
    }

    // --------------------------------------------------------
    // Evaluate objective
    // --------------------------------------------------------

    auto hedging_engine = DSO::HedgingEngine(cfg.product_price, control_intervals);
    hedging_engine.bind(gridspec);
    auto objective_linear = DSO::MCHedgeObjective(
        cfg.n_paths,
        product,
        *linear_controller,
        hedging_engine,
        risk,
        *feature_extractor
    );

    DSO::BatchSpec batch;
    batch.batch_index = 0;
    batch.first_path = 0;
    batch.n_paths = cfg.n_paths;
    batch.rng_offset = 0;

    DSO::EvalContext ctx(std::make_unique<DSO::RNGStream>(cfg.seed));
    ctx.device = torch::kCPU;
    ctx.dtype = torch::kFloat32;
    ctx.perf = nullptr;
    ctx.training = false;

    auto linear_loss = objective_linear.loss(simulated, batch, ctx);

    std::cout << "Linear controller loss = " << linear_loss.item<double>() << "\n";

    auto objective_trained = DSO::MCHedgeObjective(
        cfg.n_paths,
        product,
        trained_controller,
        hedging_engine,
        risk,
        trained_feature_extractor
    );

    auto trained_loss = objective_trained.loss(simulated, batch, ctx);
    std::cout << "Trained controller loss = " << trained_loss.item<double>() << "\n";
}

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
        std::cout << "Paths         : " << cfg.n_paths << "\n";
        std::cout << "Hedge freq    : " << cfg.hedge_freq << "\n";
        std::cout << "Risk          : " << cfg.risk_name << "\n";
        std::cout << "Learning rate : " << cfg.learning_rate << "\n";
        std::cout << "Iterations    : " << cfg.training_iters << "\n\n";

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

        model->eval();
        for (auto& p : model->parameters()) p.requires_grad_(false);

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
        model->init(gridspec);

        // ----------------------------------------------------
        // Controller
        // ----------------------------------------------------

        // auto feature_extractor = RecurrentOptionFeatureExtractor(product);
        auto feature_extractor = HestonOptionFeatureExtractor(product);
        for (auto& p : feature_extractor->parameters()) p.requires_grad_(true);
        auto controller = DSO::LinearController(feature_extractor->feature_dim());
        for (auto& p : controller->parameters()) p.requires_grad_(true);

        // ----------------------------------------------------
        // Hedging Engine
        // ----------------------------------------------------

        auto hedging_engine = DSO::HedgingEngine(cfg.product_price, intervals);
        hedging_engine.bind(gridspec);

        // ----------------------------------------------------
        // Risk
        // ----------------------------------------------------

        auto risk = make_risk(cfg);

        // ----------------------------------------------------
        // Objective + Trainer
        // ----------------------------------------------------

        auto objective = DSO::MCHedgeObjective(
            cfg.n_paths,
            product,
            *controller,
            hedging_engine,
            *risk,
            *feature_extractor
        );

        std::vector<torch::Tensor> params;

        auto control_params = controller->parameters();
        auto feat_params = feature_extractor->parameters();

        params.insert(params.end(), control_params.begin(), control_params.end());
        params.insert(params.end(), feat_params.begin(), feat_params.end());

        auto optim = DSO::Adam(
            torch::optim::Adam(
                params,
                torch::optim::AdamOptions(cfg.learning_rate)
            )
        );
        auto mc_config = DSO::MonteCarloExecutor::Config(cores, cfg.batch_size, cfg.seed, false);
        auto trainer =
            DSO::MonteCarloGradientTrainer(
                {mc_config, cfg.n_paths },
                model.ptr(),
                product,
                objective,
                optim
            );

        // ----------------------------------------------------
        // Training Loop
        // ----------------------------------------------------

        std::cout << "\nSTART TRAINING\n";

        double lr = cfg.learning_rate;
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < cfg.training_iters; ++iter) {

            auto loss = optim.step(trainer);

            lr *= cfg.lr_decay;
            lr = std::max(lr, cfg.min_lr);
            optim.set_lr(lr);

            if ((iter + 1) % 10 == 0) {
                std::cout << "iter=" << iter + 1
                          << " loss="
                          << loss.item<double>()
                          << "\n";
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::cout << "TRAINING FINISHED\n";
        std::cout << "Duration=" 
        << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) * 1e-3 << "s\n";

        // ----------------------------------------------------
        // Print trained params
        // ----------------------------------------------------

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

        // ----------------------------------------------------
        // Benchmark
        // ----------------------------------------------------

        DSO::BatchSpec batch;
        batch.batch_index = 0;
        batch.first_path = 0;
        batch.n_paths = cfg.n_paths;
        batch.rng_offset = 0;

        auto ctx = DSO::EvalContext(std::make_unique<DSO::RNGStream>(cfg.seed));
        auto simulated = model->simulate_batch(batch, ctx);

        benchmark_against_linear(
            simulated,
            product,
            *controller.ptr(),
            control_times,
            intervals,
            gridspec,
            *risk,
            *feature_extractor.ptr(),
            cfg
        );

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
