#include "dso.hpp"
#include <vector>
#include <string>
#include <torch/torch.h>

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
            DSO::MonteCarloExecutor::Config mc_config;
            size_t n_paths;
            std::vector<std::tuple<std::string, std::string>> second_order_derivatives;
        };

        MonteCarloValuation(
            Config config,
            std::shared_ptr<DSO::StochasticModelImpl> model
        )
        : config_(std::move(config))
        , model_(model)
        , executor_(config_.mc_config) {
            build_second_order_structure_();
        }

        ValueResult value(const DSO::Product& product) {

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
            if (config_.second_order_derivatives.empty()) return;

            std::vector<std::string> names;
            for (const auto& p : model_->named_parameters()) {
                names.push_back(p.key());
            }

            const size_t n = names.size();
            std::unordered_map<size_t, SecondOrderGroup> tmp;

            size_t flat_index = 0;

            for (const auto& req : config_.second_order_derivatives) {

                const auto& name_i = std::get<0>(req);
                const auto& name_j = std::get<1>(req);

                size_t i = n;
                size_t j = n;

                for (size_t k = 0; k < n; ++k) {
                    if (names[k] == name_i) i = k;
                    if (names[k] == name_j) j = k;
                }

                TORCH_CHECK(i < n && j < n, "Invalid second-order parameter name: ", name_i, ", ", name_j);

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
            const DSO::Product& product,
            size_t b,
            size_t first_path,
            size_t batch_n,
            uint64_t epoch_rng_offset,
            DSO::EvalContext& eval_ctx
        ) {

            DSO::BatchSpec batch;
            batch.batch_index = b;
            batch.first_path = first_path;
            batch.n_paths = batch_n;
            batch.rng_offset = epoch_rng_offset;

            eval_ctx.device = torch::kCPU;
            eval_ctx.dtype = torch::kFloat32;
            eval_ctx.training = true;

            auto params = model_->parameters();
            const bool need_second = !second_groups_.empty();

            auto simulated = model_->simulate_batch(batch, eval_ctx);
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
        std::shared_ptr<DSO::StochasticModelImpl> model_;
        DSO::MonteCarloExecutor executor_;
};

int main() {
    const size_t cores = std::thread::hardware_concurrency();
    const size_t num_threads = cores;
    std::cout << "cores=" << cores << "\n";
    torch::set_num_threads(1);
    torch::set_num_interop_threads(1);
    constexpr size_t n_paths = 1ULL << 20;
    constexpr size_t batch_size = 1ULL << 13;
    auto mc_config = DSO::MonteCarloExecutor::Config(
        num_threads,
        batch_size,
        /*seed=*/42,
        /*collect_perf*/false
    );
    double maturity = 1.0;
    double strike = 100.0;
    // auto product = DSO::AsianCallOption(maturity, strike, 252);
    auto product = DSO::EuropeanCallOption(maturity, strike);
    double s0 = 100.0;
    double sigma = 0.20;
    auto model = DSO::BlackScholesModel(
        DSO::BlackScholesModelImpl::Config({s0, sigma}, false)
    );
    DSO::SimulationGridSpec gridspec;
    gridspec.include_t0 = product.include_t0();
    for (auto t : product.time_grid()) {
        gridspec.time_grid.push_back(t);
    }
    model->init(gridspec);

    std::vector<std::tuple<std::string, std::string>> second_order_derivatives = {
        {"s0", "s0"}, 
        {"s0", "sigma"},
        {"sigma", "sigma"}
    };
    auto valuator = MonteCarloValuation(
        MonteCarloValuation::Config(mc_config, n_paths, second_order_derivatives),
        model.ptr()
    );
    auto start = std::chrono::high_resolution_clock::now();
    auto value = valuator.value(product);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        end - start
    ).count();
    std::cout << "Valuation duration=" << duration << "ms\n";
    std::cout << "Valuation results (ignoring risk free rate r=0):\n";
    std::cout << "value=" << value.value << "\n";

    size_t i = 0;
    for (const auto& named_param : model->named_parameters()) {
        const auto& name = named_param.key();
        std::cout << "dV/d" << name << "=" << value.gradient[i] << "\n";
        ++i;
    }

    for (size_t j = 0; j < second_order_derivatives.size(); ++j) {
        std::cout << "d^2V/"
                << "d" << std::get<0>(second_order_derivatives[j])
                << "d" << std::get<1>(second_order_derivatives[j])
                << "=" << value.second_order_derivatives[j] << "\n";
    }
}