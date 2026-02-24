#pragma once
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

#include "control/controller.hpp"
#include "features/feature_extractor.hpp"
#include "products/product.hpp"

namespace DSO {

class LinearHedgeControllerImpl final : public ControllerImpl {
public:

    LinearHedgeControllerImpl(
        std::shared_ptr<FeatureExtractorImpl> feature_extractor
    ) {
        feature_extractor_ = register_module("feature_extractor", feature_extractor);
        auto opt = torch::TensorOptions().dtype(torch::kFloat32);
        w_ = register_parameter("w", torch::rand({feature_extractor_->feature_dim()}, opt));
        b_ = register_parameter("b", torch::rand({1}, opt));
    }

    torch::Tensor forward(
        const MarketView& mv,
        const BatchSpec& batch,
        const EvalContext& ctx
    ) override {
        torch::Tensor x = feature_extractor_->features(mv, batch, ctx);
        torch::Tensor hedge = torch::matmul(x, w_) + b_;
        return hedge;
    }

private:
    std::shared_ptr<FeatureExtractorImpl> feature_extractor_;

    torch::Tensor w_;
    torch::Tensor b_;
    std::vector<double> time_grid_;
};
TORCH_MODULE(LinearHedgeController);
} // namespace DSO
