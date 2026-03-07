#pragma once
#include "dso.hpp"
#include "risk_factory.hpp"
#include "experiment_context.hpp"
#include "logged_hedging_engine.hpp"
#include "eval_metrics.hpp"
#include "logging.hpp"
#include <torch/torch.h>
#include <string>


inline void compute_moments(
    const torch::Tensor& x,
    double& mean,
    double& std,
    double& skew,
    double& kurtosis
) {
    auto x_cpu = x.detach().to(torch::kCPU);

    mean = x_cpu.mean().item<double>();

    auto centered = x_cpu - mean;
    auto var = centered.pow(2).mean().item<double>();
    std = std::sqrt(var);

    if (std > 1e-12) {
        auto m3 = centered.pow(3).mean().item<double>();
        auto m4 = centered.pow(4).mean().item<double>();

        skew = m3 / std::pow(std, 3);
        kurtosis = m4 / std::pow(std, 4);
    } else {
        skew = 0.0;
        kurtosis = 0.0;
    }
}

inline void compute_tail_metrics(
    const torch::Tensor& pnl,
    double alpha,
    double& var_alpha,
    double& cvar_alpha
) {
    // loss = -PnL
    auto losses = (-pnl).detach().to(torch::kCPU);

    auto sorted = std::get<0>(losses.sort());

    int64_t N = sorted.size(0);
    int64_t k = static_cast<int64_t>(alpha * N);

    k = std::max<int64_t>(1, std::min<int64_t>(k, N - 1));

    var_alpha = sorted[k].item<double>();
    cvar_alpha = sorted.slice(0, k, N).mean().item<double>();
}

inline EvalMetrics compute_eval_metrics(
    const LoggedHedgingResult& result,
    double alpha
) {
    EvalMetrics m;
    m.alpha = alpha;

    // ---- PnL moments ----
    compute_moments(
        result.pnl,
        m.mean_pnl,
        m.std_pnl,
        m.skew_pnl,
        m.kurtosis_pnl
    );

    m.mse = result.pnl.pow(2).mean().item<double>();

    // ---- Tail risk ----
    compute_tail_metrics(
        result.pnl,
        alpha,
        m.var_alpha,
        m.cvar_alpha
    );

    // ---- Turnover stats ----
    double dummy;
    compute_moments(
        result.total_turnover,
        m.mean_turnover,
        m.std_turnover,
        dummy,
        dummy
    );

    // ---- Transaction cost stats ----
    compute_moments(
        result.total_transaction_cost,
        m.mean_transaction_cost,
        m.std_transaction_cost,
        dummy,
        dummy
    );

    // ---- Wealth stats ----
    compute_moments(
        result.terminal_wealth,
        m.mean_terminal_wealth,
        m.std_terminal_wealth,
        dummy,
        dummy
    );

    return m;
}

EvalMetrics eval_hedge_parameters(ExperimentContext& experiment_ctx) {
    torch::NoGradGuard no_grad;
    auto& product = experiment_ctx.product;
    auto& model = experiment_ctx.model;
    auto& feature_extractor = experiment_ctx.feature_extractor;
    auto& controller = experiment_ctx.controller;
    auto& control_times = experiment_ctx.control_times;
    auto& cfg = experiment_ctx.config;

    product->to(cfg.device);
    model->to(cfg.device);
    feature_extractor->to(cfg.device);
    controller->to(cfg.device);

    model->eval();
    feature_extractor->eval();
    controller->eval();
    for (auto& p : model->parameters()) p.requires_grad_(false);
    for (auto& p : feature_extractor->parameters()) p.requires_grad_(false);
    for (auto& p : controller->parameters()) p.requires_grad_(false);

    DSO::ControlIntervals intervals;
    intervals.start_times.assign(
        control_times.begin(),
        control_times.end() - 1
    );
    intervals.end_times.assign(
        control_times.begin() + 1,
        control_times.end()
    );

    auto hedging_engine = LoggedHedgingEngine(cfg.product_price, intervals, cfg.transaction_cost_rate);
    auto master_grid = DSO::merge_time_grids(control_times, product->time_grid());

    DSO::SimulationGridSpec gridspec;
    gridspec.include_t0 = product->include_t0() || control_times.front() < 1e-12;
    gridspec.time_grid = master_grid;
    model->bind(gridspec);
    product->bind(gridspec);
    hedging_engine.bind(gridspec);

    DSO::BatchSpec batch;
    batch.batch_index = 0;
    batch.first_path = 0;
    batch.n_paths = cfg.n_eval_paths;
    batch.rng_offset = 0;

    DSO::EvalContext ctx(std::make_unique<DSO::RNGStream>(cfg.eval_seed));
    ctx.device = cfg.device;

    auto simulated = model->simulate_batch(batch, ctx);
    auto hedge_result = hedging_engine.run(
        simulated,
        *product,
        *controller,
        *feature_extractor
    );
    
    auto metrics = compute_eval_metrics(hedge_result, cfg.cvar_alpha);
    metrics.risk_name = cfg.risk_name;
    metrics.alpha = cfg.cvar_alpha;
    metrics.n_paths = cfg.n_eval_paths;
    metrics.hedging_frequency = cfg.hedge_freq;
    metrics.transaction_cost_rate = cfg.transaction_cost_rate;
    return metrics;
}