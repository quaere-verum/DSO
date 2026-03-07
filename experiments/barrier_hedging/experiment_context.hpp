#pragma once
#include "dso.hpp"
#include "config.hpp"
#include "feature_extractors.hpp"
#include <vector>
#include <memory>
#include <iostream>

struct ExperimentContext {
    std::shared_ptr<DSO::ProductImpl> product;
    std::shared_ptr<DSO::StochasticModelImpl> model;
    std::shared_ptr<DSO::FeatureExtractorImpl> feature_extractor;
    std::shared_ptr<DSO::ControllerImpl> controller;
    std::vector<double> control_times;
    ExperimentConfig config;
};

ExperimentContext create_context(const ExperimentConfig& cfg) {
    ExperimentContext ctx;
    std::cout << "EXPERIMENT CONTEXT INITIALISED\n";
    std::vector<double> monitoring_grid = DSO::make_time_grid(cfg.maturity, cfg.maturity / (double)cfg.n_time_steps, true);
    std::cout << "MONITORING GRID CREATED\n";
    auto product = DSO::DownAndOutCallOption(cfg.maturity, cfg.strike, cfg.barrier, monitoring_grid, cfg.softplus_beta, cfg.barrier_beta);
    std::cout << "PRODUCT CREATED\n";
    if (cfg.variance_mode == FeatureVarianceMode::USE_INSTANTANEOUS_VARIANCE) {
        auto feature_extractor = HestonBarrierFeatureExtractor(*product);
        ctx.feature_extractor = feature_extractor.ptr();
    } else {
        auto feature_extractor = RecurrentBarrierFeatureExtractor(*product);
        ctx.feature_extractor = feature_extractor.ptr();
    }
    std::cout << "FEATURE EXTRACTOR CREATED\n";
    auto model = DSO::HestonModel(DSO::HestonModelImpl::Config(
        DSO::HestonModelParameters(
            cfg.s0,
            cfg.v0,
            cfg.kappa,
            cfg.theta,
            cfg.xi,
            cfg.rho
        ),
        false
    ));
    std::cout << "MODEL CREATED\n";
    auto controller = DSO::MlpController(DSO::MlpControllerImpl::Config(ctx.feature_extractor->feature_dim(), cfg.hidden_sizes));
    std::cout << "CONTROLLER CREATED\n";
    std::vector<double> control_times = DSO::make_time_grid(cfg.maturity, 1.0 / cfg.hedge_freq, true);
    std::cout << "CONTROL TIMES CREATED\n";
    ctx.config = cfg;
    ctx.product = product.ptr();
    ctx.control_times = control_times;
    ctx.controller = controller.ptr();
    ctx.model = model.ptr();
    return ctx;
}