#include <torch/torch.h>
#include <vector>
#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_map>
#include "simulation/monte_carlo.hpp"
#include "models/stochastic_model.hpp"
#include "control/controller.hpp"
#include "core/stochastic_program.hpp"

namespace DSO {

struct ValueResult {
    double value;
    std::vector<double> gradient;
    std::vector<double> second_order_derivatives;
};

struct ValuationAccumResult {
    double value_sum = 0.0;
    std::vector<double> grad_sum;
    std::vector<double> second_sum;
    size_t n_paths = 0;

    void init(size_t n_params, size_t n_second) {
        if (grad_sum.empty())
            grad_sum.assign(n_params, 0.0);
        if (second_sum.empty())
            second_sum.assign(n_second, 0.0);
    }

    void merge(ValuationAccumResult other) {
        if (!other.grad_sum.empty()) {
            if (grad_sum.empty())
                grad_sum = std::move(other.grad_sum);
            else {
                for (size_t i = 0; i < grad_sum.size(); ++i)
                    grad_sum[i] += other.grad_sum[i];
            }
        }

        if (!other.second_sum.empty()) {
            if (second_sum.empty())
                second_sum = std::move(other.second_sum);
            else {
                for (size_t i = 0; i < second_sum.size(); ++i)
                    second_sum[i] += other.second_sum[i];
            }
        }

        value_sum += other.value_sum;
        n_paths += other.n_paths;
    }
};

class MonteCarloValuation {
    public:
        struct Config {
            MonteCarloExecutor::Config mc_config;
            size_t n_paths;
            std::vector<std::tuple<std::string, std::string>> second_order_derivatives;
        };

        MonteCarloValuation(
            Config config,
            StochasticModel& model
        )
        : config_(std::move(config))
        , model_(model)
        , executor_(config_.mc_config) {
            set_second_order_requests_();
            TORCH_CHECK(model_.mode() == ModelEvalMode::VALUATION, "MonteCarloValuation: model must be set to valuation mode using set_mode(mode)");
        }

        ValueResult value(const Product& product) {
            auto acc = executor_.run<ValuationAccumResult>(
                config_.n_paths,
                [&](size_t b, size_t first_path,
                    size_t batch_n, DSO::EvalContext& ctx)
                {
                    return batch_fn_(
                        product,
                        b,
                        first_path,
                        batch_n,
                        0,
                        ctx
                    );
                }
            );

            const double inv_n = 1.0 / static_cast<double>(acc.n_paths);

            ValueResult out;

            out.value = acc.value_sum * inv_n;

            out.gradient.resize(acc.grad_sum.size());
            for (size_t i = 0; i < acc.grad_sum.size(); ++i)
                out.gradient[i] = acc.grad_sum[i] * inv_n;

            if (!acc.second_sum.empty()) {
                out.second_order_derivatives.resize(acc.second_sum.size());
                for (size_t i = 0; i < acc.second_sum.size(); ++i)
                    out.second_order_derivatives[i] = acc.second_sum[i] * inv_n;
            }

            return out;
        }

    private:
        void set_second_order_requests_() {
            auto names = model_.parameter_names();
            size_t n = names.size();

            second_order_requests_.clear();
            second_order_requests_.resize(n);

            for (auto& pair : config_.second_order_derivatives) {
                const auto& name_i = std::get<0>(pair);
                const auto& name_j = std::get<1>(pair);

                size_t i = n;
                size_t j = n;

                for (size_t k = 0; k < n; ++k) {
                    if (names[k] == name_i) i = k;
                    if (names[k] == name_j) j = k;
                }

                TORCH_CHECK(i < n && j < n, "MonteCarloValuation: invalid parameter name");

                second_order_requests_[i].push_back(j);
                second_order_flat_.push_back({i,j});
            }
            for (size_t k = 0; k < second_order_flat_.size(); ++k) {
                auto [i, j] = second_order_flat_[k];
                grouped_[i].push_back({j, k});
            }
        }

        ValuationAccumResult batch_fn_(
            const Product& product,
            size_t b,
            size_t first_path,
            size_t batch_n,
            uint64_t epoch_rng_offset,
            DSO::EvalContext& eval_ctx
        ) {
            BatchSpec batch;
            batch.batch_index = b;
            batch.first_path = first_path;
            batch.n_paths = batch_n;
            batch.rng_offset = epoch_rng_offset;

            eval_ctx.device = torch::kCPU;
            eval_ctx.dtype = torch::kFloat32;
            eval_ctx.training = true;

            auto params = model_.parameters();
            const bool need_second = !config_.second_order_derivatives.empty();

            torch::Tensor simulated = model_.simulate_batch(batch, eval_ctx);
            torch::Tensor payoffs = product.compute_smooth_payoff(simulated);
            torch::Tensor value = payoffs.sum();

            std::vector<torch::Tensor> grads = torch::autograd::grad(
                    {value},
                    params,
                    {},
                    need_second,
                    need_second,
                    /*allow_unused=*/true
                );

            ValuationAccumResult out;
            out.init(params.size(), config_.second_order_derivatives.size());

            {
                torch::NoGradGuard no_grad;

                out.value_sum = value.detach().item<double>();

                for (size_t i = 0; i < grads.size(); ++i) {
                    if (grads[i].defined())
                        out.grad_sum[i] += grads[i].detach().item<double>();
                }

                if (need_second) {

                    size_t group_counter = 0;
                    const size_t n_groups = grouped_.size();

                    for (auto& [i, vec] : grouped_) {

                        const bool retain = (++group_counter < n_groups);

                        if (vec.size() == 1) {
                            size_t j = vec[0].first;
                            size_t out_idx = vec[0].second;

                            auto second = torch::autograd::grad(
                                    {grads[i]},
                                    {params[j]},
                                    {},
                                    retain,
                                    false,
                                    true
                                );

                            if (second[0].defined())
                                out.second_sum[out_idx] += second[0].detach().item<double>();
                        }
                        else {
                            auto second =
                                torch::autograd::grad(
                                    {grads[i]},
                                    params,
                                    {},
                                    retain,
                                    false,
                                    true
                                );

                            for (auto& [j, out_idx] : vec) {
                                if (second[j].defined())
                                    out.second_sum[out_idx] +=
                                        second[j].detach().item<double>();
                            }
                        }
                    }
                }
                out.n_paths = batch_n;
            }
            return out;
        }

    private:
        std::vector<std::vector<size_t>> second_order_requests_;
        std::vector<std::pair<size_t,size_t>> second_order_flat_;
        std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> grouped_;

        Config config_;
        StochasticModel& model_;
        MonteCarloExecutor executor_;
};

} // namespace DSO
