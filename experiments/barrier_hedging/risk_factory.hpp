#pragma once
#include "config.hpp"
#include <memory>
#include <iostream>
#include "dso.hpp"

std::unique_ptr<DSO::RiskMeasureImpl> make_risk(const ExperimentConfig& cfg) {
    if (cfg.risk_name == "cvar") {
        return std::make_unique<DSO::CVaRRiskImpl>(cfg.cvar_alpha);
    }
    if (cfg.risk_name == "mse") {
        return std::make_unique<DSO::MeanSquareRiskImpl>();
    }
    throw std::invalid_argument("Invalid risk: " + cfg.risk_name);
}