#pragma once
#include "features/feature_extractor.hpp"
#include "products/product.hpp"

namespace DSO {
class OptionFeatureExtractorImpl final : public FeatureExtractorImpl {
    public:
        OptionFeatureExtractorImpl(const Option& option) 
        : option_(option) {
            auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
            strike_inv_ = torch::tensor(1.0 / option_.strike(), opts);
            tau_ = torch::tensor(option_.maturity(), opts);
        }

        torch::Tensor features(
            const MarketView& mv,
            const BatchSpec& batch,
            const EvalContext& ctx
        ) override {
            auto out = torch::empty({(int64_t)batch.n_paths, 2}, torch::TensorOptions().dtype(ctx.dtype).device(ctx.device));
            const auto S = mv.S_t;

            auto col0 = out.select(1, 0);
            torch::log_out(col0, S * strike_inv_);
            out.select(1, 1).copy_(tau_ - mv.t);

            return out;
        }

        int64_t feature_dim() const override {return 2;};
    private:
        const Option& option_;
        torch::Tensor strike_inv_;
        torch::Tensor tau_;
};
TORCH_MODULE(OptionFeatureExtractor);
}