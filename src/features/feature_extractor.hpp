#pragma once
#include <torch/torch.h>
#include "products/product.hpp"
#include "core/threading.hpp"
#include <optional>

namespace DSO {

struct SimulationState {
    torch::Tensor spot;
    torch::Tensor spot_previous;
    torch::Tensor variance;
    torch::Tensor short_rate;
    torch::Tensor hidden_state;
    double t;
    double t_next;
};

struct FeatureExtractorResult {
    torch::Tensor features;
    torch::Tensor hidden_state;
};

class FeatureExtractorImpl : public torch::nn::Module {
    public:
        virtual ~FeatureExtractorImpl() = default;

        virtual FeatureExtractorResult forward(const SimulationState& state) const = 0;
        virtual const std::optional<size_t> hidden_state_dim() const = 0;
        virtual const size_t feature_dim() const = 0;

};
}
