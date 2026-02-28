#pragma once
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

#include "control/controller.hpp"
#include "features/feature_extractor.hpp"
#include "products/product.hpp"

namespace DSO {

class LinearControllerImpl final : public ControllerImpl {
public:

    LinearControllerImpl(size_t feature_dim) : feature_dim_(feature_dim) {
        auto opt = torch::TensorOptions().dtype(torch::kFloat32);
        w_ = register_parameter("w", torch::rand({(int64_t)feature_dim}, opt));
        b_ = register_parameter("b", torch::rand({1}, opt));
    }

    torch::Tensor forward(const torch::Tensor& features) const override {
        torch::Tensor action = torch::matmul(features, w_) + b_;
        return action;
    }

    const size_t feature_dim() const override { return feature_dim_; }

private:
    size_t feature_dim_;
    torch::Tensor w_;
    torch::Tensor b_;
};
TORCH_MODULE(LinearController);
} // namespace DSO
