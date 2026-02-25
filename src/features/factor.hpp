#pragma once
#include <cinttypes>

namespace DSO {
enum class FactorType : uint8_t {
    Spot = 0,
    LogSpot = 1,
    Numeraire = 2,
    DiscountFactor = 3,
    ShortRate = 4,
    Variance = 5,
    FXRate = 6,
    // Path functionals:
    RunningAverage = 7,
    RunningMax = 8,
    RunningMin = 9,
    BarrierHit = 10
};
} // namespace DSO