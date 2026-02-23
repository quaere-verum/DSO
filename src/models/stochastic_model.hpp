#pragma once
#include <torch/torch.h>
#include "products/product.hpp"
#include "core/threading.hpp"
#include "control/controller.hpp"
#include <vector>
#include <string>

namespace DSO {
enum class ModelEvalMode: uint8_t {
    VALUATION = 1,
    CALIBRATION = 2,
    HEDGING = 3,
    UNINITIATLISED = 4
};
class StochasticModel {
    public:
        virtual ~StochasticModel() = default;
        virtual torch::Tensor simulate_batch(
            const BatchSpec& batch,
            const EvalContext& ctx,
            Controller* controller = nullptr
        ) = 0;
        virtual void init(const SimulationGridSpec& spec) = 0;
        virtual std::vector<torch::Tensor>& parameters() = 0;
        virtual const std::vector<std::string>& parameter_names() const = 0;
        virtual const std::vector<DSO::FactorType>& factors() const = 0;
        virtual const ModelEvalMode mode() const = 0;
    };

} // namespace DSO
