#pragma once
#include "config.hpp"
#include "dso.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>


DSO::MlpController linear_regression_benchmark(
    const DSO::Option& product,
    DSO::StochasticModelImpl& model,
    const DSO::FeatureExtractorImpl& feature_extractor,
    const std::vector<double>& control_times,
    const ExperimentConfig& cfg
) {
    int64_t batch_size = cfg.n_paths;
    DSO::ControlIntervals control_intervals;
    control_intervals.start_times.assign(
        control_times.begin(),
        control_times.end() - 1
    );
    control_intervals.end_times.assign(
        control_times.begin() + 1,
        control_times.end()
    );

    auto master_grid = DSO::merge_time_grids(control_times, product.time_grid());

    DSO::SimulationGridSpec gridspec;
    gridspec.include_t0 = product.include_t0() || control_times.front() < 1e-12;
    gridspec.time_grid = master_grid;
    auto opt = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(cfg.device);

    DSO::BatchSpec batch;
    batch.batch_index = 0;
    batch.first_path = 0;
    batch.n_paths = cfg.n_paths;
    batch.rng_offset = 0;

    auto ctx = DSO::EvalContext(std::make_unique<DSO::RNGStream>(cfg.seed));
    ctx.device = cfg.device;

    auto simulated = model.simulate_batch(batch, ctx);

    auto payoff = product.compute_payoff(simulated);
    auto premium = torch::full({(int64_t)cfg.n_paths}, cfg.product_price, opt);

    auto control_indices = DSO::bind_to_grid(control_intervals, gridspec.time_grid);

    torch::Tensor X = torch::zeros({(int64_t)cfg.n_paths, (int64_t)feature_extractor.feature_dim() + 1}, opt);
    torch::Tensor c = torch::ones({(int64_t)cfg.n_paths, 1}, opt);
    
    {
        torch::NoGradGuard no_grad;

        DSO::SimulationState state;
        state.spot_previous = simulated.spot.select(1, 0);
        auto hidden_state_dim = feature_extractor.hidden_state_dim();
        if (hidden_state_dim) state.hidden_state = torch::zeros({(int64_t)batch_size, (int64_t) *hidden_state_dim}, simulated.spot.options());

        for (size_t k = 0; k < control_intervals.n_intervals(); ++k) {

            int64_t t0 = control_indices.start_idx[k];
            int64_t t1 = control_indices.end_idx[k];

            state.spot = simulated.spot.select(1, t0);

            if (simulated.variance.defined()) state.variance = simulated.variance.select(1, t0);
            if (simulated.short_rate.defined()) state.short_rate = simulated.short_rate.select(1, t0);

            state.t = control_intervals.start_times[k];
            state.t_next = control_intervals.end_times[k];

            auto fe_output = feature_extractor.forward(state);
            auto S0 = simulated.spot.select(1, t0);
            auto S1 = simulated.spot.select(1, t1);
            auto dS = S1 - S0;

            X += dS.view({(int64_t)batch_size, 1}) * torch::cat({c, fe_output.features}, 1);
            

            state.hidden_state = fe_output.hidden_state;
            state.spot_previous = state.spot;
        }
    }
    
    auto y = payoff.to(torch::kFloat32) - premium;

    // --------------------------------------------------------
    // Solve linear regression
    // --------------------------------------------------------

    auto XtX = torch::matmul(X.t(), X);
    auto Xty = torch::matmul(X.t(), y.unsqueeze(1));
    auto w = torch::linalg_solve(XtX, Xty).squeeze();

    // --------------------------------------------------------
    // Construct linear controller with fitted weights
    // --------------------------------------------------------
    auto linear_controller = DSO::MlpController(DSO::MlpControllerImpl::Config(feature_extractor.feature_dim(), {}));
    linear_controller->to(cfg.device);
    for (auto& param : linear_controller->named_parameters()) {
        auto& name = param.key();
        auto& tensor = param.value();
        if (name == "controller.0.bias") {
            tensor.data().copy_(w[0]);
        } else {
            tensor.data().copy_(w.slice(0, 1));
        }
    }
    return linear_controller;
}