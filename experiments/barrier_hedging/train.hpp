#pragma once
#include "experiment_context.hpp"
#include "dso.hpp"
#include <torch/torch.h>
#include "risk_factory.hpp"

void train_hedge_parameters(ExperimentContext& experiment_ctx) {
    auto& product = experiment_ctx.product;
    auto& model = experiment_ctx.model;
    auto& feature_extractor = experiment_ctx.feature_extractor;
    auto& controller = experiment_ctx.controller;
    auto& control_times = experiment_ctx.control_times;
    auto& cfg = experiment_ctx.config;
    auto risk = make_risk(cfg);

    product->to(cfg.device);
    model->to(cfg.device);
    feature_extractor->to(cfg.device);
    controller->to(cfg.device);
    risk->to(cfg.device);

    model->eval();
    for (auto& p : model->parameters()) p.requires_grad_(false);
    
    DSO::ControlIntervals intervals;
    intervals.start_times.assign(
        control_times.begin(),
        control_times.end() - 1
    );
    intervals.end_times.assign(
        control_times.begin() + 1,
        control_times.end()
    );

    auto hedging_engine = DSO::HedgingEngine(cfg.product_price, intervals, cfg.transaction_cost_rate, cfg.train_with_smooth_payoff);

    auto master_grid = DSO::merge_time_grids(control_times, product->time_grid());

    DSO::SimulationGridSpec gridspec;
    gridspec.include_t0 = product->include_t0() || control_times.front() < 1e-12;
    gridspec.time_grid = master_grid;
    model->bind(gridspec);
    product->bind(gridspec);
    hedging_engine.bind(gridspec);

    for (auto& p : feature_extractor->parameters()) p.requires_grad_(true);
    for (auto& p : controller->parameters()) p.requires_grad_(true);
    for (auto& p : risk->parameters()) p.requires_grad_(true);

    auto objective = DSO::MCHedgeObjective(
        *product,
        *controller,
        hedging_engine,
        *risk,
        *feature_extractor
    );

    std::vector<torch::Tensor> params;

    auto control_params = controller->parameters();
    auto feat_params = feature_extractor->parameters();
    auto risk_params = risk->parameters();

    params.insert(params.end(), control_params.begin(), control_params.end());
    params.insert(params.end(), feat_params.begin(), feat_params.end());
    params.insert(params.end(), risk_params.begin(), risk_params.end());

    auto optim = DSO::Adam(
        torch::optim::Adam(
            params,
            torch::optim::AdamOptions(cfg.learning_rate)
        )
    );

    auto mc_config = DSO::MonteCarloExecutor::Config(cfg.n_threads, cfg.batch_size, cfg.seed, false, cfg.device);
    auto trainer =
        DSO::MonteCarloGradientTrainer(
            { mc_config, cfg.n_train_paths, cfg.device },
            *model,
            *product,
            objective,
            optim
        );

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
    std::cout << "Duration=" 
    << static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()) * 1e-3 << "s\n";
    return;
}