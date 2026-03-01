#pragma once
#include <string>
#include <cinttypes>
#include <unordered_map>
#include <iostream>
#include <sstream>

std::vector<size_t> parse_hidden_sizes(const std::string& s) {
    std::vector<size_t> sizes;
    std::stringstream ss(s);
    std::string item;
    // Splits by comma or space
    while (std::getline(ss, item, ',')) {
        if (!item.empty()) {
            sizes.push_back(std::stoull(item));
        }
    }
    return sizes;
}

struct ExperimentConfig {
    size_t n_threads = 16;
    size_t n_paths = 1ULL << 20;
    size_t batch_size = 1ULL << 14;
    int64_t seed = 123;
    int64_t eval_seed = 456;
    double hedge_freq = 12.0;         // annual re-hedges
    std::string risk_name = "cvar";
    double risk_alpha = 0.95;         // CVaR level
    double risk_lambda = 1.0;         // entropic or mean-var param
    double learning_rate = 1e-1;
    double lr_decay = 0.985;
    double min_lr = 1e-3;
    int training_iters = 100;
    std::vector<size_t> hidden_sizes = {};

    // Fixed product parameters
    double maturity = 1.0;
    double strike = 100.0;
    double product_price = 7.97;
};

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

    if (args.count("--hidden"))
        cfg.hidden_sizes = parse_hidden_sizes(args["--hidden"]);

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