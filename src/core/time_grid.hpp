#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <torch/torch.h>

namespace DSO {

constexpr double TIME_EPS = 1e-12;

inline std::vector<double> make_time_grid(double maturity, double dt, bool include_maturity = false) {
    TORCH_CHECK(maturity > 0.0, "make_time_grid: maturity must be > 0");
    TORCH_CHECK(dt > 0, "make_time_grid: dt must be > 0");

    std::vector<double> grid;

    // Reserve approximate size to avoid reallocations
    const size_t n = static_cast<size_t>(std::ceil(maturity / dt));
    grid.reserve(n + 1);

    double t = 0.0;

    while (t < maturity - TIME_EPS) {
        grid.push_back(t);
        t += dt;
    }
    if (grid.empty() || std::abs(grid.front()) > TIME_EPS) grid.insert(grid.begin(), 0.0);
    if (include_maturity && std::abs(grid.back() - maturity) > TIME_EPS) grid.insert(grid.end(), maturity);
    return grid;
}


inline std::vector<double> merge_time_grids(
    const std::vector<double>& a,
    const std::vector<double>& b
) {
    std::vector<double> merged;
    merged.reserve(a.size() + b.size());

    size_t i = 0;
    size_t j = 0;

    auto push_unique = [&](double t) {
        if (merged.empty() || std::abs(merged.back() - t) > TIME_EPS) merged.push_back(t);
    };

    while (i < a.size() && j < b.size()) {
        double ta = a[i];
        double tb = b[j];

        if (ta < tb - TIME_EPS) {
            push_unique(ta);
            ++i;
        } else if (tb < ta - TIME_EPS) {
            push_unique(tb);
            ++j;
        } else {
            push_unique(ta);
            ++i;
            ++j;
        }
    }

    while (i < a.size()) push_unique(a[i++]);
    while (j < b.size()) push_unique(b[j++]);
    return merged;
}

} // namespace DSO
