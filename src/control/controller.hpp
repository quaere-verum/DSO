#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include "core/threading.hpp"
#include "features/feature_extractor.hpp"

namespace DSO {

class Controller {
public:
    virtual ~Controller() = default;
    virtual torch::Tensor action(
        const MarketView& mv,
        const Product& product,
        const BatchSpec& batch,
        const EvalContext& ctx
    ) = 0;
    virtual std::vector<torch::Tensor> parameters() = 0;
    virtual const std::vector<std::string>& parameter_names() const = 0;
    virtual void set_training(bool training) = 0;
};

} // namespace DSO
