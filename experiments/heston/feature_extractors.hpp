#pragma once
#include "dso.hpp"
#include <torch/torch.h>
#include <optional>

class RecurrentOptionFeatureExtractorImpl final : public DSO::FeatureExtractorImpl {

public:
    RecurrentOptionFeatureExtractorImpl(const DSO::Option& option)
        : option_(option)
    {
        auto opts = torch::TensorOptions()
                        .dtype(torch::kFloat32);

        strike_inv_ = 1.0 / option_.strike();
        maturity_ = register_buffer("maturity", torch::tensor({(float)option_.maturity()}, opts));

        alpha_raw_ = register_parameter("alpha_raw", torch::zeros({1}, opts));
        beta_ = register_parameter("beta", torch::zeros({1}, opts));
        bias_ = register_parameter("bias", torch::zeros({1}, opts));
    }

    DSO::FeatureExtractorResult forward(const DSO::SimulationState& state) const override {

        const auto& S      = state.spot;           // (batch,)
        const auto& S_prev = state.spot_previous;  // (batch,)
        const auto& h_prev = state.hidden_state;   // (batch,1)
        const auto& t      = state.t;
        const auto& t_next = state.t_next;

        const int64_t n_paths = S.size(0);

        auto logS      = torch::log(S);
        auto logS_prev = torch::log(S_prev);
        auto dlogS     = logS - logS_prev;

        auto dt = (t_next - t);
        auto sq_ret = (dlogS * dlogS) / dt;
        sq_ret = sq_ret.unsqueeze(1);

        // ---- Stable EWMA update ----
        auto alpha = torch::sigmoid(alpha_raw_);

        torch::Tensor next_hidden =
            alpha * h_prev
            + beta_ * sq_ret
            + bias_;

        // ---- Features ----
        auto log_moneyness = torch::log(S * strike_inv_).unsqueeze(1);
        auto tau = (maturity_ - t).expand({n_paths, 1});

        torch::Tensor features = torch::cat({log_moneyness, tau, next_hidden}, 1);

        DSO::FeatureExtractorResult out;
        out.features     = features;
        out.hidden_state = next_hidden;

        return out;
    }

    const size_t feature_dim() const override { return 3; }

    const std::optional<size_t> hidden_state_dim() const override { return 1; }

private:
    const DSO::Option& option_;

    double strike_inv_;
    torch::Tensor maturity_;

    torch::Tensor alpha_raw_;  // decay (sigmoid)
    torch::Tensor beta_;       // return weight
    torch::Tensor bias_;       // offset
};
TORCH_MODULE(RecurrentOptionFeatureExtractor);

class HestonOptionFeatureExtractorImpl final : public DSO::FeatureExtractorImpl {

public:
    HestonOptionFeatureExtractorImpl(const DSO::Option& option)
        : option_(option)
    {
        auto opts = torch::TensorOptions()
                        .dtype(torch::kFloat32);

        strike_inv_ = 1.0 / option_.strike();
        maturity_ = register_buffer("maturity", torch::tensor({(float)option_.maturity()}, opts));
    }

    DSO::FeatureExtractorResult forward(const DSO::SimulationState& state) const override {

        const auto& S      = state.spot;           // (batch,)
        const auto& v      = state.variance;
        const auto& t      = state.t;
        const auto& t_next = state.t_next;

        const int64_t n_paths = S.size(0);

        // ---- Features ----
        auto log_moneyness = torch::log(S * strike_inv_).unsqueeze(1);
        auto tau = (maturity_ - t).expand({n_paths, 1});

        torch::Tensor features = torch::cat({log_moneyness, tau, v.unsqueeze(1)}, 1);

        DSO::FeatureExtractorResult out;
        out.features = features;
        return out;
    }

    const size_t feature_dim() const override { return 3; }

    const std::optional<size_t> hidden_state_dim() const override { return std::nullopt; }

private:
    const DSO::Option& option_;

    double strike_inv_;
    torch::Tensor maturity_;
};
TORCH_MODULE(HestonOptionFeatureExtractor);