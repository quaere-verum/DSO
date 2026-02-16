#pragma once
#include <torch/torch.h>
#include <cstdint>

namespace DSO {
class PCG32 {
    public:
        PCG32(uint64_t seed, uint64_t seq) {
            seed_rng(seed, seq);
        };

        std::unique_ptr<PCG32> clone() const {return std::make_unique<PCG32>(*this);}

        void seed_rng(uint64_t seed, uint64_t seq = 1) {
            state_ = 0;
            inc_ = (seq << 1u) | 1u;
            next_uint();
            state_ += seed;
            next_uint();
        }

        void advance(uint64_t delta) {
            uint64_t acc_mul = 1ULL;
            uint64_t acc_inc = 0u;

            uint64_t cur_mul = multiplier_;
            uint64_t cur_inc = inc_;

            while (delta > 0) {
                if (delta & 1ULL) {
                    acc_mul = mul64_(acc_mul, cur_mul);
                    acc_inc = add64_(mul64_(acc_inc, cur_mul), cur_inc);
                }
                cur_inc = mul64_(cur_inc, add64_(cur_mul, 1u));
                cur_mul = mul64_(cur_mul, cur_mul);

                delta >>= 1;
            }

            state_ = add64_(mul64_(state_, acc_mul), acc_inc);
        }

        uint32_t next_uint() {
            uint64_t oldstate = state_;
            state_ = add64_(mul64_(oldstate, multiplier_), inc_);

            uint32_t xorshifted = static_cast<uint32_t>(
                ((oldstate >> 18u) ^ oldstate) >> 27u
            );
            uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);

            return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
        };

        float next_float() {
            return (static_cast<float>(next_uint()) + 0.5f) * inv_uint32_;
        };

    private:
        static inline uint64_t mul64_(uint64_t a, uint64_t b) {
        #if defined(__SIZEOF_INT128__)
            return (uint64_t)((__uint128_t)a * (__uint128_t)b);
        #else
            return a * b;
        #endif
        }

        static inline uint64_t add64_(uint64_t a, uint64_t b) {
            return a + b;
        }

    private:
        uint64_t state_;
        uint64_t inc_;

        static constexpr uint64_t multiplier_ = 6364136223846793005ULL;
        static constexpr float inv_uint32_ = 1.0f / 4294967296.0f;
};
} // namespace DSO