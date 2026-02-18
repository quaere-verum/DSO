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

class FeatureExtractor {
    public:
        virtual ~FeatureExtractor() = default;

        virtual torch::Tensor features(
            const MarketView& mv,
            const Product& product,
            const BatchSpec& batch,
            const EvalContext& ctx
        ) = 0;

        virtual int64_t feature_dim() const = 0;
};
}

