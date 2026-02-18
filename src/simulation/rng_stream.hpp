#pragma once
#include <cstdint>
#include "simulation/pcg32.hpp"
#include <torch/torch.h>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include <algorithm>

#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x) (x)
#define UNLIKELY(x) (x)
#endif


// TODO: Vectorisation

namespace DSO {
struct NormalVR {
    bool antithetic = true;
    bool moment_match = true;
};

struct ZigguratTables {
    static constexpr int kN = 128;
    alignas(64) std::array<uint32_t, kN> kn; // align for cache friendliness
    alignas(64) std::array<float, kN> wn;
    alignas(64) std::array<float, kN> fn;

    ZigguratTables() { init(); }

    void init() {
        // ... (Same initialization logic as your code) ...
        const double R  = 3.442619855899;
        const double VN = 9.91256303526217e-3;
        double dn = R, tn = R;
        double q  = VN / std::exp(-0.5 * dn * dn);
        const double m1 = 2147483648.0;

        kn[0] = static_cast<uint32_t>((dn / q) * m1);
        kn[1] = 0;
        wn[0] = static_cast<float>(q / m1);
        wn[kN-1] = static_cast<float>(dn / m1);
        fn[0] = 1.0f;
        fn[kN-1] = static_cast<float>(std::exp(-0.5 * dn * dn));

        for (int i = kN - 2; i >= 1; --i) {
            dn = std::sqrt(-2.0 * std::log(VN / dn + std::exp(-0.5 * dn * dn)));
            kn[i + 1] = static_cast<uint32_t>((dn / tn) * m1);
            tn = dn;
            fn[i] = static_cast<float>(std::exp(-0.5 * dn * dn));
            wn[i] = static_cast<float>(dn / m1);
        }
    }
};
static const ZigguratTables g_ziggurat_tables;

class RNGStream {
    public:
        RNGStream(uint64_t seed, uint64_t stream_id) 
        : engine_(seed, stream_id)
        , base_seed_(seed) {};

        void seed_rng(uint64_t seed, uint64_t seq) {
            base_seed_ = seed;
            engine_.seed_rng(seed, seq);
        }

        void seed_path(uint64_t path_id) {
            engine_.seed_rng(base_seed_, path_id);
        }

        std::unique_ptr<RNGStream> clone() const {return std::make_unique<RNGStream>(*this);}

        inline void fill_uniform(float* data, size_t n) {
            for (size_t i = 0; i < n; ++i) data[i] = engine_.next_float();
        }

        inline void fill_normal(float* data, size_t n, float mean = 0.0, float sigma = 1.0) {
            for (size_t i = 0; i < n; ++i) {
                float z = normal_inline_();
                data[i] = mean + sigma * z;
            }
        }

        inline void fill_normal_block(
            float* out, size_t n_paths, size_t n_dim,
            uint64_t first_path, uint64_t rng_offset,
            const NormalVR& vr, float mean, float sigma
        ) {
            const size_t N = n_paths * n_dim;

            auto gen_row = [&](size_t row, float* row_ptr) {
                const uint64_t path_id = first_path + row;
                engine_.seed_rng(base_seed_, path_id + rng_offset);
                for (size_t j = 0; j < n_dim; ++j) {
                    row_ptr[j] = normal_inline_();
                }
            };

            if (vr.antithetic) {
                const size_t half = n_paths / 2;
                for (size_t i = 0; i < half; ++i) {
                    float* a = out + i * n_dim;
                    float* b = out + (i + half) * n_dim;
                    gen_row(i, a);
                    for (size_t j = 0; j < n_dim; ++j) b[j] = -a[j];
                }
                if (n_paths % 2) {
                    gen_row(n_paths - 1, out + (n_paths - 1) * n_dim);
                }
            } else {
                for (size_t i = 0; i < n_paths; ++i) gen_row(i, out + i * n_dim);
            }

            if (vr.moment_match) moment_match_inplace_(out, N);

            if (mean != 0.0f || sigma != 1.0f) {
                for (size_t k = 0; k < N; ++k) out[k] = mean + sigma * out[k];
            }
        }

    private:
        static inline void moment_match_inplace_(float* x, size_t n) {
            double sum = 0.0;
            for (size_t i = 0; i < n; ++i) sum += x[i];
            const double mean = sum / (double)n;

            double s2 = 0.0;
            for (size_t i = 0; i < n; ++i) {
                const double d = (double)x[i] - mean;
                s2 += d * d;
            }

            const double var = s2 / (double)n;
            const double inv_std = (var > 0.0) ? (1.0 / std::sqrt(var)) : 1.0;

            for (size_t i = 0; i < n; ++i) {
                x[i] = (float)(((double)x[i] - mean) * inv_std);
            }
        }

        inline float normal_inline_() {
            const auto& T = g_ziggurat_tables;

            while (true) {
                const uint32_t u = engine_.next_uint();
                const int32_t hz = static_cast<int32_t>(u);
                const uint32_t iz = u & (ZigguratTables::kN - 1);

                // Fast Path: LIKELY helps branch prediction
                if (LIKELY(std::abs(hz) < T.kn[iz])) {
                    return hz * T.wn[iz];
                }

                // Slow Path (Tail or Strip boundary)
                if (iz == 0) {
                    float x, y;
                    do {
                        x = -std::log(engine_.next_float()) * 0.2904764f; // 1/3.4426...
                        y = -std::log(engine_.next_float());
                    } while (y + y < x * x);
                    return (hz > 0) ? (3.4426198f + x) : -(3.4426198f + x);
                } else {
                    const float x = hz * T.wn[iz];
                    const float u_flt = engine_.next_float();
                    const float fx = T.fn[iz] + u_flt * (T.fn[iz - 1] - T.fn[iz]);
                    if (fx < std::exp(-0.5f * x * x)) {
                        return x;
                    }
                }
            }
        }

        uint64_t base_seed_;
        DSO::PCG32 engine_;
};
} // namespace DSO
