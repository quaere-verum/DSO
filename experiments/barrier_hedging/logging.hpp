#pragma once
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include "eval_metrics.hpp"
#include "config.hpp"

inline std::string to_string_prec(double x, int precision = 10) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << x;
    return oss.str();
}

inline std::string to_json(const EvalMetrics& m) {

    std::ostringstream oss;
    oss << "{\n";

    // --- metadata ---
    oss << "  \"risk_name\": \"" << m.risk_name << "\",\n";
    oss << "  \"alpha\": " << to_string_prec(m.alpha) << ",\n";
    oss << "  \"n_paths\": " << m.n_paths << ",\n";
    oss << "  \"hedging_frequency\": " << to_string_prec(m.hedging_frequency) << ",\n";
    oss << "  \"transaction_cost_rate\": " << to_string_prec(m.transaction_cost_rate) << ",\n\n";

    // --- pnl stats ---
    oss << "  \"mean_pnl\": " << to_string_prec(m.mean_pnl) << ",\n";
    oss << "  \"std_pnl\": " << to_string_prec(m.std_pnl) << ",\n";
    oss << "  \"skew_pnl\": " << to_string_prec(m.skew_pnl) << ",\n";
    oss << "  \"kurtosis_pnl\": " << to_string_prec(m.kurtosis_pnl) << ",\n";
    oss << "  \"mse\": " << to_string_prec(m.mse) << ",\n\n";

    // --- tail ---
    oss << "  \"var_alpha\": " << to_string_prec(m.var_alpha) << ",\n";
    oss << "  \"cvar_alpha\": " << to_string_prec(m.cvar_alpha) << ",\n\n";

    // --- trading ---
    oss << "  \"mean_turnover\": " << to_string_prec(m.mean_turnover) << ",\n";
    oss << "  \"std_turnover\": " << to_string_prec(m.std_turnover) << ",\n";
    oss << "  \"mean_transaction_cost\": " << to_string_prec(m.mean_transaction_cost) << ",\n";
    oss << "  \"std_transaction_cost\": " << to_string_prec(m.std_transaction_cost) << ",\n\n";

    // --- wealth ---
    oss << "  \"mean_terminal_wealth\": " << to_string_prec(m.mean_terminal_wealth) << ",\n";
    oss << "  \"std_terminal_wealth\": " << to_string_prec(m.std_terminal_wealth) << "\n";

    oss << "}\n";

    return oss.str();
}

inline void save_json(
    const EvalMetrics& m,
    const std::string& filepath
) {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filepath);
    }

    file << to_json(m);
    file.close();
}

inline std::string config_to_json(const ExperimentConfig& cfg) {
    std::ostringstream oss;

    oss << "{\n";
    oss << "  \"risk_name\": \"" << cfg.risk_name << "\",\n";
    oss << "  \"cvar_alpha\": " << cfg.cvar_alpha << ",\n";
    oss << "  \"learning_rate\": " << cfg.learning_rate << ",\n";
    oss << "  \"training_iters\": " << cfg.training_iters << ",\n";
    oss << "  \"n_paths\": " << cfg.n_train_paths << ",\n";
    oss << "  \"transaction_cost_rate\": " << cfg.transaction_cost_rate << ",\n";
    oss << "  \"strike\": " << cfg.strike << ",\n";
    oss << "  \"barrier\": " << cfg.barrier << ",\n";
    oss << "  \"xi\": " << cfg.xi << ",\n";
    oss << "  \"rho\": " << cfg.rho << ",\n";
    oss << "  \"softplus_beta\": " << cfg.softplus_beta << ",\n";
    oss << "  \"barrier_beta\": " << cfg.barrier_beta << "\n";
    oss << "}\n";

    return oss.str();
}

inline void save_config_json(
    const ExperimentConfig& cfg,
    const std::string& filepath
) {
    std::ofstream file(filepath);
    if (!file.is_open())
        throw std::runtime_error("Could not open config file: " + filepath);

    file << config_to_json(cfg);
}