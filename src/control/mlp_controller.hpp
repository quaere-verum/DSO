#pragma once
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

#include "control/controller.hpp"
#include "features/feature_extractor.hpp"
#include "products/product.hpp"

namespace DSO {
class MlpControllerImpl final : public ControllerImpl {
public:
    struct Config {
        size_t feature_dim;
        std::vector<size_t> hidden_sizes = {};
    };
    MlpControllerImpl(const Config& config) : config_(config) {
        auto controller = torch::nn::Sequential();
        size_t in_channels = config_.feature_dim;

        for (const auto& hidden_size : config_.hidden_sizes) {
            controller->push_back(torch::nn::Linear(in_channels, hidden_size));
            controller->push_back(torch::nn::ReLU());
            in_channels = hidden_size;
        }
        controller->push_back(torch::nn::Linear(in_channels, 1));
        controller_ = register_module("controller", controller);
    }

    torch::Tensor forward(const torch::Tensor& features) const override {
        return controller_->forward(features).squeeze(1);
    }

    const size_t feature_dim() const override { return config_.feature_dim; }


private:
    const Config& config_;
    std::shared_ptr<torch::nn::SequentialImpl> controller_;
};
TORCH_MODULE(MlpController);
} // namespace DSO
