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

struct ValueAccumResult {
    torch::Tensor value_sum = torch::tensor({0.0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    size_t n_paths = 0;

    void merge(ValueAccumResult other) {
        value_sum = value_sum + other.value_sum;
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
        }

        ValueResult value(const Product& product) {
            model_.set_mode(DSO::ModelEvalMode::VALUATION);
            auto params = model_.parameters();
            ValueAccumResult total_value =
                executor_.run<ValueAccumResult>(
                    config_.n_paths,
                    [&](size_t b, size_t first_path,
                        size_t batch_n, DSO::EvalContext& ctx
                    ) { return batch_fn_(product, b, first_path, batch_n, 0, ctx); }
                );

            torch::Tensor value = total_value.value_sum / torch::tensor({(float)total_value.n_paths}, total_value.value_sum.options());

            const bool need_second = !config_.second_order_derivatives.empty();

            auto grads = torch::autograd::grad(
                {value},
                params,
                {},
                /*retain_graph=*/need_second,
                /*create_graph=*/need_second
            );

            ValueResult out;
            out.value = value.item<double>();
            out.gradient.reserve(grads.size());
            for (auto& g : grads)
                out.gradient.push_back(
                    g.defined() ? g.item<double>() : 0.0
                );

            if (need_second) {

                const size_t n_params = params.size();
                std::vector<double> results(
                    config_.second_order_derivatives.size(),
                    0.0
                );

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
                            /*retain_graph=*/retain,
                            /*create_graph=*/false,
                            /*allow_unused=*/true
                        );

                        results[out_idx] = second[0].defined() ? second[0].item<double>() : 0.0;
                    }
                    else {
                        auto second = torch::autograd::grad(
                            {grads[i]},
                            params,
                            {},
                            /*retain_graph=*/retain,
                            /*create_graph=*/false,
                            /*allow_unused=*/true
                        );

                        for (auto& [j, out_idx] : vec) {
                            results[out_idx] = second[j].defined() ? second[j].item<double>() : 0.0;
                        }
                    }
                }

                out.second_order_derivatives = std::move(results);
            }

            return out;
        }

    private:

        ValueAccumResult batch_fn_(
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

            torch::Tensor simulated =
                model_.simulate_batch(batch, eval_ctx);

            auto payoffs = torch::empty({(int64_t)batch_n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

            product.compute_smooth_payoff(simulated, payoffs);

            ValueAccumResult out;
            out.value_sum = payoffs.sum();
            out.n_paths = batch_n;
            return out;
        }

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

    private:
        std::vector<std::vector<size_t>> second_order_requests_;
        std::vector<std::pair<size_t,size_t>> second_order_flat_;
        std::unordered_map<size_t, std::vector<std::pair<size_t, size_t>>> grouped_;

        Config config_;
        StochasticModel& model_;
        MonteCarloExecutor executor_;
};

} // namespace DSO
