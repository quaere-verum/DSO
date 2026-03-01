#pragma once
#include "dso.hpp"
#include "config.hpp"
#include <memory>

std::unique_ptr<DSO::RiskMeasureImpl> make_risk(const ExperimentConfig& cfg) {

    if (cfg.risk_name == "mse")
        return std::make_unique<DSO::MeanSquareRiskImpl>();

    if (cfg.risk_name == "cvar")
        return std::make_unique<DSO::CVaRRiskImpl>(cfg.risk_alpha);

    if (cfg.risk_name == "entropic")
        return std::make_unique<DSO::EntropicRiskImpl>(cfg.risk_lambda);

    if (cfg.risk_name == "meanvar")
        return std::make_unique<DSO::MeanVarianceRiskImpl>(cfg.risk_lambda);

    throw std::invalid_argument("Unknown risk: " + cfg.risk_name);
}