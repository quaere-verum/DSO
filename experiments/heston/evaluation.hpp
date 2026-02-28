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
        , executor_(config_.mc_config) {}

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

            eval_ctx.device = torch::kCPU;
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
};

double eval_hedge_parameters(
    const DSO::Product& product,
    DSO::StochasticModelImpl& model,
    DSO::FeatureExtractorImpl& feature_extractor,
    DSO::ControllerImpl& controller,
    DSO::RiskMeasure& risk,
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

    for (auto& p : feature_extractor.parameters()) p.requires_grad_(true);
    for (auto& p : controller.parameters()) p.requires_grad_(true);

    auto hedging_engine = DSO::HedgingEngine(cfg.product_price, intervals);
    hedging_engine.bind(gridspec);

    auto objective = DSO::MCHedgeObjective(
        product,
        controller,
        hedging_engine,
        risk,
        feature_extractor
    );

    auto evaluator = MonteCarloLoss(
        MonteCarloLoss::Config(DSO::MonteCarloExecutor::Config(cfg.n_threads, cfg.batch_size, cfg.eval_seed), cfg.n_paths),
        model,
        objective
    );

    auto loss = evaluator.loss();
    return loss;
}