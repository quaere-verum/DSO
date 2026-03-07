#pragma once
#include <string>
#include <cinttypes>
#include <unordered_map>
#include <iostream>
#include <sstream>
#include <torch/torch.h>
#include <vector>
#include "dso.hpp"

std::vector<size_t> parse_hidden_sizes(const std::string& s) {
    std::vector<size_t> sizes;
    std::string token;
    std::stringstream ss;

    for (char c : s) {
        if (c == ',' || std::isspace(c))
            ss << ' ';
        else
            ss << c;
    }

    std::string normalized = ss.str();
    std::stringstream ns(normalized);

    while (ns >> token) {
        sizes.push_back(std::stoull(token));
    }

    return sizes;
}

enum class FeatureVarianceMode {
    USE_INSTANTANEOUS_VARIANCE,
    USE_LEARNED_REPRESENTATION
};

struct ExperimentConfig {

    // =========================
    // Monte Carlo
    // =========================
    size_t n_threads        = 16;
    size_t n_train_paths    = 1ULL << 20;
    size_t n_eval_paths     = 1ULL << 20;
    size_t batch_size       = 1ULL << 14;
    int64_t seed            = 123;
    int64_t eval_seed       = 456;

    // =========================
    // Time Grid
    // =========================
    double maturity         = 1.0;
    size_t n_time_steps     = 252;      // simulation grid resolution
    double hedge_freq       = 252.0;    // hedges per year

    // =========================
    // Product (Lookback)
    // =========================
    double strike                 = 100.0;
    double barrier                = 80.0;
    bool train_with_smooth_payoff = true;
    double softplus_beta          = 10.0;
    double barrier_beta           = 20.0;
    double product_price;                  // Determined through MC valuation after init

    // =========================
    // Heston Model
    // =========================
    double s0               = 100.0;
    double v0               = 0.04;
    double kappa            = 2.0;
    double theta            = 0.04;
    double xi               = 1.0;
    double rho              = -0.7;

    // =========================
    // Policy Architecture
    // =========================
    std::vector<size_t> hidden_sizes = {};
    FeatureVarianceMode variance_mode = FeatureVarianceMode::USE_INSTANTANEOUS_VARIANCE;

    // =========================
    // Risk Objective
    // =========================
    std::string risk_name   = "cvar";  // cvar, mse
    double cvar_alpha       = 0.95;

    // =========================
    // Optimisation
    // =========================
    double learning_rate    = 0.075;
    double lr_decay         = 0.99;
    double min_lr           = 1e-4;
    int training_iters      = 200;

    // =========================
    // Market Frictions
    // =========================
    double transaction_cost_rate = 10e-4;

    // =========================
    // Infrastructure
    // =========================
    torch::Device device = torch::kCUDA;
    std::string output_dir = "logs";
};

ExperimentConfig parse_args(int argc, char* argv[]) {

    ExperimentConfig cfg;
    std::unordered_map<std::string, std::string> args;

    for (int i = 1; i < argc - 1; i += 2)
        args[argv[i]] = argv[i + 1];

    auto get = [&](const std::string& key) -> std::string {
        return args.count(key) ? args[key] : "";
    };

    if (args.count("--train-paths"))
        cfg.n_train_paths = std::stoull(get("--train-paths"));

    if (args.count("--eval-paths"))
        cfg.n_eval_paths = std::stoull(get("--eval-paths"));

    if (args.count("--batch-size"))
        cfg.batch_size = std::stoull(get("--batch-size"));

    if (args.count("--maturity"))
        cfg.maturity = std::stod(get("--maturity"));

    if (args.count("--strike"))
        cfg.strike = std::stod(get("--strike"));

    if (args.count("--barrier"))
        cfg.barrier = std::stod(get("--barrier"));

    if (args.count("--train-with-real-payoff"))
        cfg.train_with_smooth_payoff = false;

    if (args.count("--time-steps"))
        cfg.n_time_steps = std::stoull(get("--time-steps"));

    if (args.count("--hedge-freq"))
        cfg.hedge_freq = std::stod(get("--hedge-freq"));

    if (args.count("--variance-mode")) {
        auto v = get("--variance-mode");
        if (v == "instant")
            cfg.variance_mode =
                FeatureVarianceMode::USE_INSTANTANEOUS_VARIANCE;
        else if (v == "learned")
            cfg.variance_mode =
                FeatureVarianceMode::USE_LEARNED_REPRESENTATION;
        else
            throw std::invalid_argument("Invalid variance-mode");
    }

    if (args.count("--hidden"))
        cfg.hidden_sizes = parse_hidden_sizes(get("--hidden"));

    if (args.count("--risk"))
        cfg.risk_name = get("--risk");

    if (args.count("--alpha"))
        cfg.cvar_alpha = std::stod(get("--alpha"));

    if (args.count("--lr"))
        cfg.learning_rate = std::stod(get("--lr"));

    if (args.count("--iters"))
        cfg.training_iters = std::stoi(get("--iters"));

    if (args.count("--device"))
        cfg.device = (get("--device") == "cuda")
            ? torch::kCUDA
            : torch::kCPU;

    if (args.count("--output"))
        cfg.output_dir = get("--output");

    // basic validation
    if (cfg.hedge_freq <= 0.0)
        throw std::invalid_argument("hedge_freq must be > 0");

    if (cfg.n_time_steps == 0)
        throw std::invalid_argument("n_time_steps must be > 0");

    if (cfg.learning_rate <= 0.0)
        throw std::invalid_argument("learning_rate must be > 0");

    return cfg;
}
