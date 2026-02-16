#pragma once

#include "core/differentiable_objective.hpp"
#include "core/stochastic_program.hpp"
#include "core/threading.hpp"

#include "models/stochastic_model.hpp"
#include "models/black_scholes.hpp"

#include "simulation/pcg32.hpp"
#include "simulation/rng_stream.hpp"
#include "simulation/monte_carlo.hpp"

#include "products/product.hpp"
#include "products/european_option.hpp"

#include "objectives/mc_price_objective.hpp"