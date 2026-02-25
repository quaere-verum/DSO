#pragma once

#include "core/differentiable_objective.hpp"
#include "core/stochastic_program.hpp"
#include "core/threading.hpp"
#include "core/time_grid.hpp"

#include "models/stochastic_model.hpp"
#include "models/black_scholes.hpp"

#include "simulation/pcg32.hpp"
#include "simulation/rng_stream.hpp"
#include "simulation/monte_carlo.hpp"

#include "products/product.hpp"
#include "products/european_option.hpp"
#include "products/asian_option.hpp"

#include "objectives/mc_calibration_objective.hpp"
#include "objectives/mc_hedge_objective.hpp"

#include "trainers/optimiser.hpp"
#include "trainers/lbfgs_wrapper.hpp"
#include "trainers/adam_wrapper.hpp"

#include "trainers/mc_gradients.hpp"

#include "control/controller.hpp"
#include "control/linear_hedge_controller.hpp"

#include "features/feature_extractor.hpp"
#include "features/option_features.hpp"

#include "hedging/hedging.hpp"

#include "risk/risk_measure.hpp"
#include "risk/mean_squared_error.hpp"
#include "risk/mean_variance.hpp"
#include "risk/entropic_risk.hpp"
#include "risk/cvar.hpp"
