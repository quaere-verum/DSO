#pragma once
#include "features/feature_extractor.hpp"
#include "products/product.hpp"
#include <optional>

namespace DSO {
class OptionFeatureExtractorImpl final : public FeatureExtractorImpl {
    public:
        OptionFeatureExtractorImpl(const Option& option) 
        : option_(option) {
            strike_inv_ = 1.0 / option_.strike();
            tau_ = option_.maturity();
        }

        FeatureExtractorResult forward(const SimulationState& state) const override {
            std::vector<torch::Tensor> features;
            features.reserve(feature_dim());
            const auto S = state.spot;
            const int64_t batch_size = S.size(0);
            features.push_back(torch::log(S * strike_inv_));
            features.push_back(torch::full({batch_size}, tau_ - state.t, S.options()));
            FeatureExtractorResult out;
            out.features = torch::stack(features, 1);
            return out;
        }

        const std::optional<size_t> hidden_state_dim() const { return std::nullopt; }
        const size_t feature_dim() const override {return 2;};

    private:
        const Option& option_;
        double strike_inv_;
        double tau_;
};
TORCH_MODULE(OptionFeatureExtractor);
}