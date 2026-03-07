#pragma once
#include <string>

struct EvalMetrics {

    // ---- configuration metadata ----
    std::string risk_name;
    double alpha = 0.0;
    int64_t n_paths = 0;
    double hedging_frequency = 0.0;
    double transaction_cost_rate = 0.0;

    // ---- PnL distribution ----
    double mean_pnl = 0.0;
    double std_pnl = 0.0;
    double skew_pnl = 0.0;
    double kurtosis_pnl = 0.0;

    double mse = 0.0;

    // ---- Tail risk ----
    double var_alpha = 0.0;
    double cvar_alpha = 0.0;

    // ---- Trading behaviour ----
    double mean_turnover = 0.0;
    double std_turnover = 0.0;

    double mean_transaction_cost = 0.0;
    double std_transaction_cost = 0.0;

    // ---- Wealth diagnostics ----
    double mean_terminal_wealth = 0.0;
    double std_terminal_wealth = 0.0;
};