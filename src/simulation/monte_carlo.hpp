#pragma once
#include "core/threading.hpp"
#include "simulation/rng_stream.hpp"
#include <tbb/task_arena.h>
#include <tbb/parallel_for.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_reduce.h>
#include <concepts>
#include <type_traits>

namespace DSO {
template<class R>
concept MergeableResult =
    std::default_initializable<R> &&
    std::movable<R> &&
    requires(R a, R b) {
        { a.merge(b) } -> std::same_as<void>;
    }
    || requires(R a, R b) {
        { a.merge(b) } -> std::same_as<R&>;
    };


template<class BatchFunc, class R>
concept BatchComputes =
    requires(BatchFunc f, size_t b, size_t first_path, size_t n, DSO::ThreadContext& ctx) {
        { f(b, first_path, n, ctx) } -> std::same_as<R>;
    };

class MonteCarloExecutor {
    public:
        struct Config {
            size_t num_threads;
            size_t batch_size;
            size_t seed;
            bool disable_torch_threads = true;
        };

        MonteCarloExecutor(
            Config config
        )
        : config_(std::move(config))
        , arena_(config_.num_threads) {
            base_rng_ = std::make_unique<RNGStream>(config_.seed, 0);
        };

        template<class Result, class BatchFunc>
        requires MergeableResult<Result> && BatchComputes<BatchFunc, Result>
        Result run(size_t total_paths, BatchFunc&& f) {
            TORCH_CHECK(total_paths > 0, "total_paths must be > 0");
            const size_t B = config_.batch_size;
            const size_t n_batches = (total_paths + B - 1) / B;

            std::vector<Result> partial_results(n_batches);
            
            // TOOD: Figure out what to do with nested parallellism (tbb + torch) -> potential oversubscription
            arena_.execute([&] {
                tbb::enumerable_thread_specific<DSO::ThreadContext> tls_ctx(
                    [&] {return DSO::ThreadContext(base_rng_->clone());}
                );

                tbb::parallel_for(size_t(0), n_batches, [&](size_t b) {
                    auto& ctx = tls_ctx.local();
                    const size_t first_path = b * B;
                    const size_t batch_n = std::min(B, total_paths - first_path);
                    partial_results[b] = std::invoke(f, b, first_path, batch_n, ctx);
                });
            });

            Result final_result = std::move(partial_results[0]);
            for (size_t i = 1; i < n_batches; ++i) {
                final_result.merge(partial_results[i]);
            }
            return final_result;
        }

    private:
        Config config_;
        std::unique_ptr<DSO::RNGStream> base_rng_;
        tbb::task_arena arena_;
};

} // namespace DSO