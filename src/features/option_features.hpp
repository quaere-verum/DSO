#pragma once
#include "features/feature_extractor.hpp"
#include "products/product.hpp"

namespace DSO {
class OptionFeatureExtractor final : public FeatureExtractor {
    public:
        torch::Tensor features(
            const MarketView& mv,
            const Product& product,
            const BatchSpec& batch,
            const EvalContext& ctx
        ) override {
            auto* opt = dynamic_cast<const Option*>(&product);
            TORCH_CHECK(opt != nullptr, "OptionFeatureExtractor requires an Option product");

            auto out = torch::empty({(int64_t)batch.n_paths, 2}, torch::TensorOptions().dtype(ctx.dtype).device(ctx.device));

            const auto S = mv.S_t;
            const float K = static_cast<float>(opt->strike());
            const float tau = static_cast<float>(opt->maturity() - mv.t);

            auto col0 = out.select(1, 0);
            col0.copy_(S);
            col0.div_(K);
            col0.log_();
            out.select(1, 1).fill_(tau);

            return out;
        }

        int64_t feature_dim() const override {return 2;};
};
}