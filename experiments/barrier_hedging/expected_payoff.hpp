#pragma once
#include "dso.hpp"
#include <vector>
#include "experiment_context.hpp"
#include "config.hpp"

void compute_expected_payoff(ExperimentContext& experiment_ctx) {
    torch::NoGradGuard no_grad;
    auto& product = experiment_ctx.product;
    auto& model = experiment_ctx.model;
    auto& control_times = experiment_ctx.control_times;
    auto& cfg = experiment_ctx.config;

    product->to(cfg.device);
    model->to(cfg.device);

    model->eval();
    for (auto& p : model->parameters()) p.requires_grad_(false);

    auto master_grid = DSO::merge_time_grids(control_times, product->time_grid());

    DSO::SimulationGridSpec gridspec;
    gridspec.include_t0 = product->include_t0() || control_times.front() < 1e-12;
    gridspec.time_grid = master_grid;
    model->bind(gridspec);
    product->bind(gridspec);

    DSO::BatchSpec batch;
    batch.batch_index = 0;
    batch.first_path = 0;
    batch.n_paths = cfg.n_eval_paths;
    batch.rng_offset = 0;

    DSO::EvalContext ctx(std::make_unique<DSO::RNGStream>(cfg.eval_seed));
    ctx.device = cfg.device;

    auto simulated = model->simulate_batch(batch, ctx);
    auto payoff = product->compute_payoff(simulated);
    cfg.product_price = payoff.mean().item<double>();
    std::cout << "ADDED PRODUCT VALUE (value=" << cfg.product_price << ")\n";
    return;
}