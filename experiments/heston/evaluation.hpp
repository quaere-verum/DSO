#pragma once
#include "config.hpp"
#include "dso.hpp"
#include <torch/torch.h>

struct LossAccumResult {
    double loss_sum = 0.0;
    size_t n_batches = 0;

    void merge(LossAccumResult other) {
        loss_sum += other.loss_sum;
        n_batches += other.n_batches;
    }
};

class MonteCarloLoss {
    public:
        struct Config {
            DSO::MonteCarloExecutor::Config mc_config;
            size_t n_paths;
        };

        MonteCarloLoss(
            Config config,
            DSO::StochasticModelImpl& model,
            DSO::StochasticProgram& objective
        )
        : config_(std::move(config))
        , model_(model)
        , objective_(objective)
        , executor_(config_.mc_config)
        , device_(config_.mc_config.device) {}

        double loss() {

            auto acc = executor_.run<LossAccumResult>(
                config_.n_paths,
                [&](
                    size_t b, 
                    size_t first_path,
                    size_t batch_n, 
                    DSO::EvalContext& ctx
                ) {
                    return batch_fn_(
                        b,
                        first_path,
                        batch_n,
                        0,
                        ctx
                    );
                }
            );

            double out = acc.loss_sum / static_cast<double>(acc.n_batches);
            return out;
        }

    private:
        LossAccumResult batch_fn_(
            size_t b,
            size_t first_path,
            size_t batch_n,
            uint64_t epoch_rng_offset,
            DSO::EvalContext& eval_ctx
        ) {

            DSO::BatchSpec batch;
            batch.batch_index = b;
            batch.first_path = first_path;
            batch.n_paths = batch_n;
            batch.rng_offset = epoch_rng_offset;

            eval_ctx.device = device_;
            eval_ctx.dtype = torch::kFloat32;
            eval_ctx.training = true;

            
            auto simulated = model_.simulate_batch(batch, eval_ctx);
            auto loss = objective_.loss(simulated, batch, eval_ctx);
            LossAccumResult out;
            out.loss_sum = loss.item<double>();
            out.n_batches = 1;
            return out;
        }

    private:
        Config config_;
        DSO::StochasticModelImpl& model_;
        DSO::StochasticProgram& objective_;
        DSO::MonteCarloExecutor executor_;
        torch::Device device_;
};

double eval_hedge_parameters(
    const DSO::Product& product,
    DSO::StochasticModelImpl& model,
    DSO::FeatureExtractorImpl& feature_extractor,
    DSO::ControllerImpl& controller,
    std::vector<double>& control_times,
    const ExperimentConfig& cfg
) {
    model.eval();
    for (auto& p : model.parameters()) p.requires_grad_(false);

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
    model.init(gridspec);

    for (auto& p : feature_extractor.parameters()) p.requires_grad_(false);
    for (auto& p : controller.parameters()) p.requires_grad_(false);

    feature_extractor.eval();
    controller.eval();

    auto hedging_engine = DSO::HedgingEngine(cfg.product_price, intervals);
    hedging_engine.bind(gridspec);


    DSO::BatchSpec batch;
    batch.batch_index = 0;
    batch.first_path = 0;
    batch.n_paths = cfg.n_paths;
    batch.rng_offset = 0;

    DSO::EvalContext ctx(std::make_unique<DSO::RNGStream>(cfg.eval_seed));
    ctx.device = cfg.device;

    auto simulated = model.simulate_batch(batch, ctx);
    auto hedge_result = hedging_engine.run(
        simulated,
        product,
        controller,
        feature_extractor
    );
    
    if (cfg.risk_name == "cvar") {
        auto pnl = hedge_result.pnl;
        torch::Tensor loss = -pnl;

        double alpha = cfg.risk_alpha;

        torch::Tensor z = torch::quantile(loss, alpha, /*dim=*/0, /*interpolation=*/"linear");
        torch::Tensor tail = torch::relu(loss - z);
        torch::Tensor cvar = z + tail.mean() / (1.0 - alpha);

        return cvar.item<double>();
    } else {
        auto risk = make_risk(cfg);
        auto loss = risk->forward(hedge_result).item<double>();
        return loss;
    }
}