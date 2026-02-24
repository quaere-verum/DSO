#pragma once
#include <torch/torch.h>
#include "products/product.hpp"
#include "core/threading.hpp"

namespace DSO {
struct MarketView {
    torch::Tensor S_t;
    int64_t t_index;
    double t;
    double t_next;
};

class FeatureExtractorImpl : public torch::nn::Module {
    public:
        virtual ~FeatureExtractorImpl() = default;

        virtual torch::Tensor features(
            const MarketView& mv,
            const BatchSpec& batch,
            const EvalContext& ctx
        ) = 0;

        virtual int64_t feature_dim() const = 0;
};
}

