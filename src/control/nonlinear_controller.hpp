#pragma once
#include <torch/torch.h>
#include <memory>
#include <string>
#include <vector>

#include "control/controller.hpp"
#include "features/feature_extractor.hpp"
#include "products/product.hpp"

namespace DSO {
class MlpPolicyImpl : public torch::nn::Module {
    public:
        virtual torch::Tensor forward(torch::Tensor features) const = 0;
        virtual const size_t feature_dim() const = 0;
};

class NonlinearControllerImpl final : public ControllerImpl {
public:

    NonlinearControllerImpl(
        std::shared_ptr<MlpPolicyImpl> policy
    ) {
        policy_ = register_module("policy", policy);
    }

    torch::Tensor forward(const torch::Tensor& features) const override {
        torch::Tensor action = policy_->forward(features);
        return action;
    }

    const size_t feature_dim() const override { return policy_->feature_dim(); }


private:
    std::shared_ptr<MlpPolicyImpl> policy_;
};
TORCH_MODULE(NonlinearController);
} // namespace DSO
