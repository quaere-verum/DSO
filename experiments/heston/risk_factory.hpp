#pragma once
#include "dso.hpp"
#include "config.hpp"
#include <memory>

std::unique_ptr<DSO::RiskMeasure> make_risk(const ExperimentConfig& cfg) {

    if (cfg.risk_name == "mse")
        return std::make_unique<DSO::MeanSquareRisk>();

    if (cfg.risk_name == "cvar")
        return std::make_unique<DSO::CVaRRisk>(cfg.risk_alpha);

    if (cfg.risk_name == "entropic")
        return std::make_unique<DSO::EntropicRisk>(cfg.risk_lambda);

    if (cfg.risk_name == "meanvar")
        return std::make_unique<DSO::MeanVarianceRisk>(cfg.risk_lambda);

    throw std::invalid_argument("Unknown risk: " + cfg.risk_name);
}