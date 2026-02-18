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
    (
        // merge from lvalue (non-const)
        requires(R a, R b) {
            { a.merge(b) } -> std::same_as<void>;
        } ||
        requires(R a, R b) {
            { a.merge(b) } -> std::same_as<R&>;
        } ||

        // merge from const lvalue
        requires(R a, const R b) {
            { a.merge(b) } -> std::same_as<void>;
        } ||
        requires(R a, const R b) {
            { a.merge(b) } -> std::same_as<R&>;
        } ||

        // merge from rvalue (move)
        requires(R a, R b) {
            { a.merge(std::move(b)) } -> std::same_as<void>;
        } ||
        requires(R a, R b) {
            { a.merge(std::move(b)) } -> std::same_as<R&>;
        }
    );



template<class BatchFunc, class R>
concept BatchComputes =
    requires(BatchFunc f, size_t b, size_t first_path, size_t n, DSO::EvalContext& ctx) {
        { f(b, first_path, n, ctx) } -> std::same_as<R>;
    };

template<class Result>
struct ReduceState {
    Result result;
    DSO::EvalContext ctx;

    explicit ReduceState(std::unique_ptr<DSO::RNGStream> rng)
        : result{}
        , ctx(std::move(rng)) {}
};

class MonteCarloExecutor {
    public:
        struct Config {
            size_t num_threads;
            size_t batch_size;
            size_t seed;
            bool collect_perf = false;
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
            TORCH_CHECK(torch::get_num_interop_threads() == 1, "set torch interop threads to 1");
            TORCH_CHECK(torch::get_num_threads() == 1, "set torch intraop threads to 1");

            const size_t B = config_.batch_size;
            const size_t n_batches = (total_paths + B - 1) / B;
            bool collect_perf = config_.collect_perf;
            Result final{};

            DSO::PerfCounters counters{};

            arena_.execute([&] {
                tbb::enumerable_thread_specific<DSO::PerfCounters> tls_perf;

                tbb::enumerable_thread_specific<DSO::EvalContext> tls_ctx([&] {
                    DSO::EvalContext ctx(base_rng_->clone());
                    if (collect_perf) {
                        ctx.perf = &tls_perf.local();
                    }
                    return ctx;
                });


                tbb::enumerable_thread_specific<Result> tls_acc(
                    [&] { return Result{}; }
                );

                const size_t grain = 1;

                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, n_batches, grain),
                    [&](const tbb::blocked_range<size_t>& r) {
                        auto& ctx = tls_ctx.local();
                        auto& acc = tls_acc.local();

                        for (size_t b = r.begin(); b != r.end(); ++b) {
                            const size_t first_path = b * B;
                            const size_t batch_n = std::min(B, total_paths - first_path);

                            Result tmp = std::invoke(f, b, first_path, batch_n, ctx);
                            acc.merge(std::move(tmp));
                        }
                    }
                );

                bool first = true;
                for (auto& a : tls_acc) {
                    if (first) { final = std::move(a); first = false; }
                    else { final.merge(std::move(a)); }
                }
                bool first_counters = true;
                for (auto& t : tls_perf) {
                    if (first_counters) { counters = std::move(t); first = false; }
                    else { counters.merge(std::move(t)); }
                }
                
            });
            if (collect_perf) DSO::print_perf(counters, total_paths);
            return final;
        }



    private:
        Config config_;
        std::unique_ptr<DSO::RNGStream> base_rng_;
        tbb::task_arena arena_;
};

} // namespace DSO