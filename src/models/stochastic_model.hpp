#pragma once
#include <torch/torch.h>
#include "products/product.hpp"
#include "core/threading.hpp"
#include "control/controller.hpp"
#include <vector>
#include <string>

namespace DSO {
class StochasticModel {
    public:
        virtual ~StochasticModel() = default;
        virtual torch::Tensor simulate_batch(
            const BatchSpec& batch,
            const EvalContext& ctx,
            Controller* controller = nullptr
        ) = 0;
        virtual void init(const SimulationGridSpec& spec) = 0;
        virtual std::vector<torch::Tensor> parameters() = 0;
        virtual const std::vector<std::string>& parameter_names() const = 0;
        virtual const std::vector<DSO::FactorType>& factors() const = 0;
        virtual void set_training(bool training) = 0;
    };

} // namespace DSO
