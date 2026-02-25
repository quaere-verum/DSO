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

void benchmark_hedge_against_linear(
    size_t n_paths,
    double product_price,
    DSO::MonteCarloExecutor::Config mc_config,
    std::shared_ptr<DSO::StochasticModelImpl> model,
    const DSO::Option& product,
    const std::shared_ptr<DSO::ControllerImpl>& trained_controller,
    const std::vector<double>& control_times,
    const DSO::ControlIntervals& control_intervals,
    const DSO::SimulationGridSpec& gridspec,
    const DSO::RiskMeasure& risk
) {
    model->eval();
    for (auto& param : model->parameters())
        param.requires_grad_(false);

    DSO::EvalContext ctx(
        std::make_unique<DSO::RNGStream>(0)
    );
    ctx.device = torch::kCPU;
    ctx.dtype = torch::kFloat32;
    ctx.training = false;

    // ---- Build batch ----
    DSO::BatchSpec batch;
    batch.batch_index = 0;
    batch.first_path = 0;
    batch.n_paths = n_paths;
    batch.rng_offset = 0;

    // ---- Simulate once ----
    auto simulated = model->simulate_batch(batch, ctx, nullptr);

    torch::Tensor payoff = product.compute_payoff(simulated);

    auto opt = torch::TensorOptions()
        .dtype(ctx.dtype)
        .device(ctx.device);

    torch::Tensor premium_tensor = torch::full({(int64_t)n_paths}, product_price, opt);

    auto control_indices = DSO::bind_to_grid(control_intervals, gridspec.time_grid);

    // ---- Linear regression features ----
    torch::Tensor A1 = torch::zeros({(int64_t)n_paths}, opt);
    torch::Tensor A2 = torch::zeros({(int64_t)n_paths}, opt);
    torch::Tensor A3 = torch::zeros({(int64_t)n_paths}, opt);

    for (size_t k = 0; k < control_intervals.n_intervals(); ++k) {

        int64_t t0_idx = control_indices.start_idx[k];
        int64_t t1_idx = control_indices.end_idx[k];

        torch::Tensor S0k = simulated.spot.select(1, t0_idx);
        torch::Tensor S1k = simulated.spot.select(1, t1_idx);

        torch::Tensor dS = S1k - S0k;

        torch::Tensor xk = torch::log(S0k / product.strike());

        double tau_k = product.maturity() - control_times[k];

        A1 += xk * dS;
        A2 += torch::full_like(dS, tau_k) * dS;
        A3 += dS;
    }

    torch::Tensor X = torch::stack({A1, A2, A3}, 1);
    torch::Tensor y = payoff.to(ctx.dtype) - premium_tensor;

    // ---- Solve closed-form linear regression ----
    torch::Tensor XtX = torch::matmul(X.t(), X);
    torch::Tensor Xty = torch::matmul(X.t(), y.unsqueeze(1));
    torch::Tensor w = torch::linalg_solve(XtX, Xty).squeeze();

    double w1 = w[0].item<double>();
    double w2 = w[1].item<double>();
    double b  = w[2].item<double>();

    std::cout << "Optimal linear hedge coefficients:\n"
              << "w1=" << w1 << "\n"
              << "w2=" << w2 << "\n"
              << "b =" << b << "\n";

    // ---- Build linear controller with fitted weights ----
    auto feature_extractor = DSO::OptionFeatureExtractor(product);
    auto linear_controller = DSO::LinearHedgeController(feature_extractor.ptr());
    for (auto& param : linear_controller->named_parameters()) {

        auto& name = param.key();
        auto& tensor = param.value();

        if (name == "b") {
            tensor.data().copy_(
                torch::tensor({(float)b}, opt)
            );
        } else {
            tensor.data().copy_(
                torch::tensor({(float)w1, (float)w2}, opt)
            );
        }
    }

    // ---- Create objective for linear controller ----
    auto hedging_engine = DSO::HedgingEngine(product_price, control_intervals);

    hedging_engine.bind(gridspec);

    auto linear_objective = DSO::MCHedgeObjective(
        n_paths,
        product,
        *linear_controller,
        hedging_engine,
        risk
    );

    auto linear_loss = linear_objective.loss(simulated, batch, ctx);
    std::cout << "Linear Controller loss = " << linear_loss.item<double>() << "\n";
    // ---- Evaluate trained controller ----
    auto trained_objective = DSO::MCHedgeObjective(
        n_paths,
        product,
        *trained_controller,
        hedging_engine,
        risk
    );

    auto trained_loss = trained_objective.loss(simulated, batch, ctx);

    std::cout << "Trained Controller loss = " << trained_loss.item<double>() << "\n";
}

void hedging(
    size_t n_paths,
    double product_price,
    DSO::MonteCarloExecutor::Config mc_config,
    std::shared_ptr<DSO::StochasticModelImpl> model,
    const DSO::Option& product
) {
    model->eval();
    for (auto& param : model->parameters()) {
        param.requires_grad_(false);
    }
    const double maturity = product.maturity();
    std::vector<double> control_times = DSO::make_time_grid(maturity, maturity / 12.0, /*include_maturity*/true);
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
    auto feature_extractor = DSO::OptionFeatureExtractor(product);
    auto controller = DSO::LinearHedgeController(
        feature_extractor.ptr()
    );
    for (auto& param : controller->parameters()) {
        param.requires_grad_(true);
    }
    auto hedging_engine = DSO::HedgingEngine(
        product_price,
        control_intervals
    );
    hedging_engine.bind(gridspec);
    model->init(gridspec);
    
    // auto risk_measure = DSO::MeanSquareRisk();
    auto risk_measure = DSO::CVaRRisk(0.95);
    auto objective = DSO::MCHedgeObjective(
        n_paths,
        product, 
        *controller,
        hedging_engine,
        risk_measure
    );
    double lr = 1e-1;
    auto optim = DSO::Adam(torch::optim::Adam(controller->parameters(), torch::optim::AdamOptions(lr)));
    auto trainer = DSO::MonteCarloGradientTrainer(
        DSO::MonteCarloGradientTrainer::Config(
            mc_config,
            n_paths
        ),
        model,
        product,
        objective,
        optim,
        controller.ptr()
    );
    std::cout << "STARTED HEDGING TRAINING\n";
    for (int iter = 0; iter < 100; ++iter) {
        torch::Tensor loss = optim.step(trainer);
        lr *= 0.98;
        optim.set_lr(lr);
        if ((iter + 1) % 10 == 0) {
            std::cout << "iter=" << iter + 1
                    << " loss=" << loss.item<float>() << "\n";
        }
    }
    std::cout << "FINISHED HEDGING TRAINING\n";
    for (const auto& named_param : controller->named_parameters()) {
        const auto& name = named_param.key();
        const auto& value = named_param.value();
        std::cout << name << "=\n" << value << "\n";
    }
    benchmark_hedge_against_linear(
        n_paths,
        product_price,
        mc_config,
        model,
        product,
        controller.ptr(),
        control_times,
        control_intervals,
        gridspec,
        risk_measure
    );
    return;
}

void calibration(
    size_t n_paths,
    double product_price,
    const DSO::MonteCarloExecutor::Config& mc_config,
    std::shared_ptr<DSO::StochasticModelImpl> model,
    const DSO::Product& product
) {
    for (auto& named_param : model->named_parameters()) {
        const auto& name = named_param.key();
        auto& param = named_param.value();
        if (name == "log_sigma") {
            param.requires_grad_(true);   // calibrate sigma
        } else {
            param.requires_grad_(false);  // freeze everything else
        }
    }
    auto objective = DSO::MCCalibrationObjective(
        product_price,
        n_paths,
        product
    );
    DSO::SimulationGridSpec gridspec;
    gridspec.time_grid = product.time_grid();
    gridspec.include_t0 = product.include_t0();
    model->init(gridspec);
    double lr = 5e-2;
    auto optim = DSO::Adam(torch::optim::Adam(model->parameters(), torch::optim::AdamOptions(lr)));
    auto trainer = DSO::MonteCarloGradientTrainer(
        DSO::MonteCarloGradientTrainer::Config(
            mc_config,
            n_paths
        ),
        model,
        product,
        objective,
        optim
    );
    std::cout << "STARTED CALIBRATION\n";
    for (int iter = 0; iter < 100; ++iter) {
        torch::Tensor loss = optim.step(trainer);
        lr *= 0.975;
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
        if (name == "log_sigma") {
            std::cout << "sigma=\n" << value.detach().exp() << "\n";
        }
    }
}


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
        double s0 = 100.0;
        double sigma_initial = 0.30;
        double product_price = 7.97; // implied vol = 20%
        auto model = DSO::BlackScholesModel(s0, sigma_initial, /*use_log_sigma*/true);
        
        calibration(n_paths, product_price, mc_config, model.ptr(), product);
        hedging(n_paths, product_price, mc_config, model.ptr(), product);

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
