#pragma once
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

#include "control/controller.hpp"
#include "features/feature_extractor.hpp"
#include "products/product.hpp"

namespace DSO {

class LinearHedgeController final : public Controller {
public:
    struct Config {
        bool squash_sigmoid = false;  // if true: hedge = sigmoid(raw)
        bool clamp = false;           // if true: clamp to [clamp_min, clamp_max]
        float clamp_min = -2.0f;
        float clamp_max =  2.0f;
    };

    LinearHedgeController(
        std::unique_ptr<FeatureExtractor> feature_extractor, 
        Config config = {}
    )
        : feature_extractor_(std::move(feature_extractor)), config_(config) {
        TORCH_CHECK(feature_extractor_ != nullptr, "LinearHedgeController: feature extractor is null");
        TORCH_CHECK(feature_extractor_->feature_dim() == 2, "LinearHedgeController: expected feature_dim()==2");

        auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);

        w_ = torch::tensor({1.0, 0.5}, opt).requires_grad_(true);
        // w_ = torch::ones({2}, opt).add_(1).requires_grad_(true);
        b_ = torch::zeros({1}, opt).add_(0.509).requires_grad_(true);

        names_ = {"hedge_w", "hedge_b"};
    }

    torch::Tensor action(
        const MarketView& mv,
        const BatchSpec& batch,
        const EvalContext& ctx
    ) override {
        torch::Tensor x = feature_extractor_->features(mv, batch, ctx);
        TORCH_CHECK(x.dim() == 2 && x.size(1) == 2, "LinearHedgeController: features must be [B,2]");
        torch::Tensor raw = torch::matmul(x, w_) + b_;
        torch::Tensor hedge = raw;
        if (config_.squash_sigmoid) {
            hedge = torch::sigmoid(hedge);
        }
        if (config_.clamp) {
            hedge = hedge.clamp(config_.clamp_min, config_.clamp_max);
        }
        return hedge;
    }

    std::vector<torch::Tensor> parameters() override {
        return {w_, b_};
    }

    const std::vector<std::string>& parameter_names() const override {
        return names_;
    }

    void set_training(bool training) override {
        w_.requires_grad_(training);
        b_.requires_grad_(training);
    }

private:
    std::unique_ptr<FeatureExtractor> feature_extractor_;
    Config config_;

    torch::Tensor w_;
    torch::Tensor b_;
    std::vector<std::string> names_;
    std::vector<double> time_grid_;
};

} // namespace DSO
