#pragma once
#include <torch/torch.h>
#include <vector>
#include <string>
#include <cstdint>
#include <utility>
#include <memory>

#include "simulation/monte_carlo.hpp"
#include "core/threading.hpp"
#include "models/stochastic_model.hpp"
#include "core/stochastic_program.hpp"
#include "products/product.hpp"
#include "trainers/optimiser.hpp"
#include "control/controller.hpp"

namespace DSO {
struct GradAccumResult {
    std::vector<torch::Tensor> grad_sum;
    double loss_sum = 0.0;
    size_t n_paths = 0;

    GradAccumResult() = default;
    GradAccumResult(GradAccumResult&&) = default;
    GradAccumResult& operator=(GradAccumResult&&) = default;

    void init_like(const std::vector<torch::Tensor>& params) {
        if (!grad_sum.empty()) return;
        grad_sum.reserve(params.size());
        for (const auto& p : params) {
            auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(p.device());
            grad_sum.emplace_back(torch::zeros_like(p, opt));
        }
    }

    void merge(GradAccumResult other) {
        if (!other.grad_sum.empty()) {
            if (grad_sum.empty()) {
                grad_sum = std::move(other.grad_sum);
            } else {
                TORCH_CHECK(grad_sum.size() == other.grad_sum.size(), "GradAccumResult: grad size mismatch");
                for (size_t i = 0; i < grad_sum.size(); ++i) {
                    grad_sum[i].add_(other.grad_sum[i]);
                }
            }
        }
        loss_sum += other.loss_sum;
        n_paths += other.n_paths;
    }
};

class MonteCarloGradientTrainer final : public GradientEvaluator {
public:
    struct Config {
        MonteCarloExecutor::Config mc_config;
        size_t n_paths;
        torch::Device device = torch::kCPU;
    };

    MonteCarloGradientTrainer(
        Config config,
        StochasticModelImpl& model,
        const Product& product,
        StochasticProgram& objective,
        Optimiser& optimiser
    )
        : config_(std::move(config))
        , model_(model)
        , product_(product)
        , objective_(objective)
        , optimiser_(optimiser)
        , mc_config_(config_.mc_config) {

        TORCH_CHECK(config_.n_paths > 0, "MonteCarloGradientTrainer: n_paths must be > 0");
        TORCH_CHECK(model_.factors() == product_.factors(), "MonteCarloGradientTrainer: model factors must match product factors");
        for (auto& group : optimiser_.param_groups()) {
            for (auto& p : group.params()) {
                if (p.requires_grad()) trainable_params_.push_back(p);
            }
        }
        TORCH_CHECK(!trainable_params_.empty(), "MonteCarloGradientTrainer: no trainable parameters found in optimiser");
    }

    torch::Tensor evaluate_and_set_grads() {
        const uint64_t epoch_rng_offset = objective_.epoch_rng_offset();
        GradAccumResult acc = mc_config_.run<GradAccumResult>(
            config_.n_paths,
            [&](size_t b, size_t first_path, size_t batch_n, DSO::EvalContext& eval_ctx) -> GradAccumResult {
                return batch_fn_(b, first_path, batch_n, epoch_rng_offset, eval_ctx);
            }
        );
        
        apply_grads_(acc);
        const double mean_loss = acc.loss_sum / static_cast<double>(acc.n_paths);
        auto opt = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
        return torch::tensor({static_cast<float>(mean_loss)}, opt);
    }

private:
    GradAccumResult batch_fn_(
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

        eval_ctx.device = config_.device;
        eval_ctx.dtype = torch::kFloat32;
        eval_ctx.training = true;

        auto* perf = eval_ctx.perf;

        auto simulated = model_.simulate_batch(batch, eval_ctx);
        
        torch::Tensor loss;
        {
            std::optional<DSO::ScopedTimer> t;
            if (perf) t.emplace(*perf, DSO::Stage::Loss);
            loss = objective_.loss(simulated, batch, eval_ctx);
        }

        std::vector<torch::Tensor> grads;
        {
            std::optional<DSO::ScopedTimer> t;
            if (perf) t.emplace(*perf, DSO::Stage::Grad);
            grads = torch::autograd::grad(
                /*outputs=*/{loss},
                /*inputs=*/trainable_params_,
                /*grad_outputs=*/{},
                /*retain_graph=*/false,
                /*create_graph=*/false,
                /*allow_unused=*/true
            );
        }
         
        GradAccumResult out;
        {
            std::optional<DSO::ScopedTimer> t;
            if (perf) t.emplace(*perf, DSO::Stage::Accum);
            out.init_like(trainable_params_);
            {
                torch::NoGradGuard no_grad;

                for (size_t i = 0; i < grads.size(); ++i) {
                    if (!grads[i].defined()) continue;
                    torch::Tensor g = grads[i].detach();
                    out.grad_sum[i].add_(g * (float)batch_n);
                }
                out.loss_sum = loss.detach().item<double>() * (double)batch_n;
                out.n_paths = batch_n;
            }
        }
        return out;
    }

    void apply_grads_(const GradAccumResult& acc) {
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < trainable_params_.size(); ++i) {
            torch::Tensor g = acc.grad_sum[i] / static_cast<double>(acc.n_paths);
            trainable_params_[i].mutable_grad() = g; 
        }
    }

private:
    Config config_;
    StochasticModelImpl& model_;
    const Product& product_;
    StochasticProgram& objective_;
    Optimiser& optimiser_;
    MonteCarloExecutor mc_config_;
    std::vector<torch::Tensor> trainable_params_;
};

} // namespace DSO
