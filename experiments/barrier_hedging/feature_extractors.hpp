#pragma once
#include "dso.hpp"
#include <torch/torch.h>
#include <optional>

class RecurrentBarrierFeatureExtractorImpl final : public DSO::FeatureExtractorImpl {

public:
    RecurrentBarrierFeatureExtractorImpl(const DSO::DownAndOutCallOptionImpl& option)
        : option_(option)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);

        maturity_ = register_buffer("maturity",
                    torch::tensor({(float)option_.maturity()}, opts));

        barrier_ = register_buffer("barrier",
                    torch::tensor({(float)option_.barrier()}, opts));
        
        strike_ = register_buffer("strike",
                    torch::tensor({(float)option_.strike()}, opts));

        alpha_raw_ = register_parameter("alpha_raw", torch::zeros({1}, opts));
        beta_      = register_parameter("beta", torch::zeros({1}, opts));
        bias_      = register_parameter("bias", torch::zeros({1}, opts));
    }

    DSO::FeatureExtractorResult forward(const DSO::SimulationState& state) override {

        const auto& S      = state.spot;
        const auto& S_min  = state.spot_cumulative_min;
        const auto& S_prev = state.spot_previous;
        const auto& h_prev = state.hidden_state;
        const auto& t      = state.t;
        const auto& t_next = state.t_next;

        const int64_t n_paths = S.size(0);

        auto logS      = torch::log(S);
        auto logS_prev = torch::log(S_prev);
        auto dlogS     = logS - logS_prev;

        auto dt = (t_next - t);

        auto sq_ret = (dlogS * dlogS) / dt;
        sq_ret = sq_ret.unsqueeze(1);

        // ---- Stable EWMA volatility proxy ----
        auto alpha = torch::sigmoid(alpha_raw_);

        torch::Tensor next_hidden =
            alpha * h_prev
            + beta_ * sq_ret
            + bias_;

        // ---- Features ----
        auto tau = (maturity_ - t).expand({n_paths,1});

        auto log_barrier_dist = torch::log(S / barrier_).unsqueeze(1);

        auto log_moneyness = torch::log(S / strike_).unsqueeze(1);

        auto alive = (S_min  > barrier_).to(S.dtype()).unsqueeze(1);

        torch::Tensor features =
            torch::cat({log_barrier_dist, log_moneyness, alive, tau, next_hidden}, 1);

        DSO::FeatureExtractorResult out;
        out.features     = features;
        out.hidden_state = next_hidden;

        return out;
    }

    const size_t feature_dim() const override { return 5; }

    const std::optional<size_t> hidden_state_dim() const override { return 1; }

private:

    const DSO::DownAndOutCallOptionImpl& option_;

    torch::Tensor maturity_;
    torch::Tensor barrier_;
    torch::Tensor strike_;

    torch::Tensor alpha_raw_;
    torch::Tensor beta_;
    torch::Tensor bias_;
};

TORCH_MODULE(RecurrentBarrierFeatureExtractor);

class HestonBarrierFeatureExtractorImpl final : public DSO::FeatureExtractorImpl {

public:
    HestonBarrierFeatureExtractorImpl(const DSO::DownAndOutCallOptionImpl& option)
        : option_(option)
    {
        auto opts = torch::TensorOptions().dtype(torch::kFloat32);

        maturity_ = register_buffer("maturity",
                    torch::tensor({(float)option_.maturity()}, opts));

        barrier_ = register_buffer("barrier",
                    torch::tensor({(float)option_.barrier()}, opts));
            
        strike_ = register_buffer("strike",
                    torch::tensor({(float)option_.strike()}, opts));
    }

    DSO::FeatureExtractorResult forward(const DSO::SimulationState& state) override {

        const auto& S = state.spot;
        const auto& S_min = state.spot_cumulative_min;
        const auto& v = state.variance;
        const auto& t = state.t;

        const int64_t n_paths = S.size(0);

        auto tau = (maturity_ - t).expand({n_paths,1});

        auto log_moneyness = torch::log(S / strike_).unsqueeze(1);
        
        auto alive = (S_min  > barrier_).to(S.dtype()).unsqueeze(1);

        auto log_barrier_dist =
            torch::log(S / barrier_).unsqueeze(1);

        torch::Tensor features =
            torch::cat({log_barrier_dist, log_moneyness, alive, tau, v.unsqueeze(1)}, 1);

        DSO::FeatureExtractorResult out;
        out.features = features;

        return out;
    }

    const size_t feature_dim() const override { return 5; }

    const std::optional<size_t> hidden_state_dim() const override { return std::nullopt; }

private:

    const DSO::DownAndOutCallOptionImpl& option_;

    torch::Tensor maturity_;
    torch::Tensor barrier_;
    torch::Tensor strike_;
};

TORCH_MODULE(HestonBarrierFeatureExtractor);