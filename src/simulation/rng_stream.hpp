#pragma once
#include <cstdint>
#include "simulation/pcg32.hpp"
#include <torch/torch.h>
#include <vector>
#include <cmath>

// TODO: Vectorisation

namespace DSO {
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

        torch::Tensor uniform(std::vector<int64_t> shape, torch::Device device) {
            TORCH_CHECK(device.is_cpu(), "RNGStream currently supports CPU tensors only.");
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(device);
            at::Tensor tensor = at::empty(shape, options);
            size_t n = tensor.numel();
            float* data = tensor.data_ptr<float>();
            fill_uniform(data, n);
            return tensor;
        }

        torch::Tensor normal(std::vector<int64_t> shape, torch::Device device, float mu = 0.0, float sigma = 1.0) {
            TORCH_CHECK(device.is_cpu(), "RNGStream currently supports CPU tensors only.");
            auto options = torch::TensorOptions()
                .dtype(torch::kFloat32)
                .device(device);
            at::Tensor tensor = at::empty(shape, options);
            size_t n = tensor.numel();
            float* data = tensor.data_ptr<float>();
            fill_normal(data, n, mu, sigma);
            return tensor;
        }

        void uniform(torch::Tensor out) {
            TORCH_CHECK(out.device().is_cpu(), "CPU only");
            TORCH_CHECK(out.dtype() == torch::kFloat32, "float32 only");
            TORCH_CHECK(out.is_contiguous(), "uniform requires contiguous tensor");
            size_t n = out.numel();
            float* data = out.data_ptr<float>();
            fill_uniform(data, n);
            return;
        }

        void normal(torch::Tensor out, float mu = 0.0, float sigma = 1.0) {
            TORCH_CHECK(out.device().is_cpu(), "CPU only");
            TORCH_CHECK(out.dtype() == torch::kFloat32, "float32 only");
            TORCH_CHECK(out.is_contiguous(), "normal requires contiguous tensor");
            size_t n = out.numel();
            float* data = out.data_ptr<float>();
            fill_normal(data, n, mu, sigma);
            return;
        }

        inline void fill_uniform(float* data, size_t n) {
            for (size_t i = 0; i < n; ++i) data[i] = engine_.next_float();
        }

        inline void fill_normal(float* data, size_t n, float mean = 0.0, float sigma = 1.0) {
            for (size_t i = 0; i < n; ++i) {
                float z = normal_();
                data[i] = mean + sigma * z;
            }
        }

    private:
        struct ZigguratTables {
            static constexpr int kN = 128;

            std::array<uint32_t, kN> kn{};
            std::array<float,    kN> wn{};
            std::array<float,    kN> fn{};

            void init() {
                // Constants from Marsaglia & Tsang for kN=128 normal ziggurat
                // R is the right-most x-coordinate of the base strip.
                const double R  = 3.442619855899;
                const double VN = 9.91256303526217e-3;

                double dn = R;
                double tn = R;
                double q  = VN / std::exp(-0.5 * dn * dn);

                // Scale is 2^31 (matches signed int32 magnitude)
                const double m1 = 2147483648.0; // 2^31

                kn[0]     = static_cast<uint32_t>((dn / q) * m1);
                kn[1]     = 0;

                wn[0]     = static_cast<float>(q / m1);
                wn[kN-1]  = static_cast<float>(dn / m1);

                fn[0]     = 1.0f;
                fn[kN-1]  = static_cast<float>(std::exp(-0.5 * dn * dn));

                for (int i = kN - 2; i >= 1; --i) {
                    dn = std::sqrt(-2.0 * std::log(VN / dn + std::exp(-0.5 * dn * dn)));
                    kn[i + 1] = static_cast<uint32_t>((dn / tn) * m1);
                    tn = dn;
                    fn[i] = static_cast<float>(std::exp(-0.5 * dn * dn));
                    wn[i] = static_cast<float>(dn / m1);
                }
            }
        };

        static const ZigguratTables& ziggurat_tables_() {
            static ZigguratTables T;
            static std::once_flag once;
            std::call_once(once, [&] { T.init(); });
            return T;
        }

        inline float normal_() {
            const auto& T = ziggurat_tables_();

            for (;;) {
                const int32_t hz = static_cast<int32_t>(engine_.next_uint());
                const uint32_t iz = static_cast<uint32_t>(hz) & (ZigguratTables::kN - 1);

                if (static_cast<uint32_t>(std::abs(hz)) < T.kn[iz]) {
                    return hz * T.wn[iz];
                }

                if (iz == 0) {
                    float x, y;
                    do {
                        x = -std::log(engine_.next_float()) / 3.442619855899f;
                        y = -std::log(engine_.next_float());
                    } while (y + y < x * x);

                    return (hz > 0) ? (3.442619855899f + x) : -(3.442619855899f + x);
                } else {
                    const float x = hz * T.wn[iz];
                    const float u = engine_.next_float();
                    const float fx = T.fn[iz] + u * (T.fn[iz - 1] - T.fn[iz]);
                    if (fx < std::exp(-0.5f * x * x)) {
                        return x;
                    }
                    // else: retry
                }
            }
        }

        uint64_t base_seed_;
        DSO::PCG32 engine_;
};
} // namespace DSO
