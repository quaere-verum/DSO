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
        if (grad_sum.empty()) {
            grad_sum.assign(n_params, 0.0);
        }
        if (second_sum.empty()) {
            second_sum.assign(n_second, 0.0);
        }
    }

    void merge(ValuationAccumResult other) {
        if (!other.grad_sum.empty()) {
            if (grad_sum.empty()) {
                grad_sum = std::move(other.grad_sum);
            } else {
                for (size_t i = 0; i < grad_sum.size(); ++i) {
                    grad_sum[i] += other.grad_sum[i];
                }
            }
        }

        if (!other.second_sum.empty()) {
            if (second_sum.empty()) {
                second_sum = std::move(other.second_sum);
            } else {
                for (size_t i = 0; i < second_sum.size(); ++i) {
                    second_sum[i] += other.second_sum[i];
                }
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
            build_second_order_structure_();
            TORCH_CHECK(model_.mode() == ModelEvalMode::VALUATION, "MonteCarloValuation: model must be in valuation mode");
        }

        ValueResult value(const Product& product) {

            auto acc = executor_.run<ValuationAccumResult>(
                config_.n_paths,
                [&](
                    size_t b, 
                    size_t first_path,
                    size_t batch_n, 
                    DSO::EvalContext& ctx
                ) {
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
                for (size_t i = 0; i < acc.second_sum.size(); ++i) {
                    out.second_order_derivatives[i] = acc.second_sum[i] * inv_n;
                }
            }

            return out;
        }

    private:
        struct SecondOrderGroup {
            size_t i;
            std::vector<size_t> js;
            std::vector<size_t> output_indices;
        };

        void build_second_order_structure_() {

            second_groups_.clear();

            if (config_.second_order_derivatives.empty())
                return;

            auto names = model_.parameter_names();
            const size_t n = names.size();

            std::unordered_map<size_t, SecondOrderGroup> tmp;

            size_t flat_index = 0;

            for (auto& req : config_.second_order_derivatives) {

                const auto& name_i = std::get<0>(req);
                const auto& name_j = std::get<1>(req);

                size_t i = n;
                size_t j = n;

                for (size_t k = 0; k < n; ++k) {
                    if (names[k] == name_i) i = k;
                    if (names[k] == name_j) j = k;
                }

                TORCH_CHECK(i < n && j < n,
                            "Invalid second-order parameter name");

                auto& group = tmp[i];
                group.i = i;
                group.js.push_back(j);
                group.output_indices.push_back(flat_index);

                ++flat_index;
            }

            for (auto& kv : tmp)
                second_groups_.push_back(std::move(kv.second));
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
            const bool need_second = !second_groups_.empty();

            torch::Tensor simulated = model_.simulate_batch(batch, eval_ctx);
            torch::Tensor payoffs = product.compute_smooth_payoff(simulated);
            torch::Tensor value = payoffs.sum();

            auto grads = torch::autograd::grad(
                {value},
                params,
                {},
                /*retain_graph=*/need_second,
                /*create_graph=*/need_second,
                /*allow_unused=*/true
            );

            ValuationAccumResult out;
            out.init(params.size(), config_.second_order_derivatives.size());

            {
                torch::NoGradGuard no_grad;

                out.value_sum = value.detach().item<double>();

                for (size_t i = 0; i < grads.size(); ++i) {
                    if (grads[i].defined()) out.grad_sum[i] += grads[i].detach().item<double>();
                }
            }

            if (need_second) {

                const size_t n_groups = second_groups_.size();
                size_t counter = 0;

                for (auto& group : second_groups_) {

                    const bool retain = (++counter < n_groups);

                    auto hess_row = torch::autograd::grad(
                        {grads[group.i]},
                        params,
                        {},
                        /*retain_graph=*/retain,
                        /*create_graph=*/false,
                        /*allow_unused=*/true
                    );

                    torch::NoGradGuard no_grad;

                    for (size_t k = 0; k < group.js.size(); ++k) {

                        size_t j = group.js[k];
                        size_t out_idx = group.output_indices[k];

                        if (hess_row[j].defined())
                            out.second_sum[out_idx] += hess_row[j].detach().item<double>();
                    }
                }
            }

            out.n_paths = batch_n;
            return out;
        }

    private:
        std::vector<SecondOrderGroup> second_groups_;

        Config config_;
        StochasticModel& model_;
        MonteCarloExecutor executor_;
};

} // namespace DSO
