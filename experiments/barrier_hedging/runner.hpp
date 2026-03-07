#pragma once
#include <filesystem>
#include <iostream>
#include "experiment_context.hpp"
#include "logging.hpp"
#include "train.hpp"
#include "eval.hpp"

class ExperimentRunner {
public:
    explicit ExperimentRunner(ExperimentContext&& ctx)
        : ctx_(std::move(ctx)) {}

    void run() {

        create_output_directory();

        log_config();

        std::cout << "\n=== START TRAINING ===\n";
        train();

        std::cout << "\n=== START EVALUATION ===\n";
        auto metrics = evaluate();

        save_metrics(metrics);
        save_model();

        std::cout << "\n=== EXPERIMENT COMPLETE ===\n";
    }

private:
    ExperimentContext ctx_;

    void create_output_directory() {
        const auto& dir = ctx_.config.output_dir;

        if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
        }
    }

    void log_config() {
        const auto& dir = ctx_.config.output_dir;
        save_config_json(
            ctx_.config,
            dir + "/config.json"
        );
    }

    void train() {
        train_hedge_parameters(ctx_);
    }

    EvalMetrics evaluate() {
        auto metrics = eval_hedge_parameters(ctx_);
        return metrics;
    }

    void save_metrics(const EvalMetrics& metrics) {
        save_json(
            metrics,
            ctx_.config.output_dir + "/metrics.json"
        );
    }

    void save_model() {
        torch::nn::ModuleHolder<DSO::FeatureExtractorImpl> holder_fe(ctx_.feature_extractor);
        torch::nn::ModuleHolder<DSO::ControllerImpl> holder_controller(ctx_.controller);
        torch::save(holder_fe, ctx_.config.output_dir + "/feature_extractor.pt");
        torch::save(holder_controller, ctx_.config.output_dir + "/controller.pt");
    }
};