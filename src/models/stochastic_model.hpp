#pragma once
#include <torch/torch.h>
#include "products/product.hpp"
#include "core/threading.hpp"
#include "control/controller.hpp"
#include <vector>
#include <string>
#include <memory>

namespace DSO {
class StochasticModelImpl : public torch::nn::Module {
    public:
        virtual ~StochasticModelImpl() = default;
        virtual torch::Tensor simulate_batch(
            const BatchSpec& batch,
            const EvalContext& ctx,
            std::shared_ptr<ControllerImpl> controller = nullptr
        ) = 0;
        virtual void init(const SimulationGridSpec& spec) = 0;
        virtual const std::vector<DSO::FactorType>& factors() const = 0;
    };
} // namespace DSO
