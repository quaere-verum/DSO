#pragma once
#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include <torch/torch.h>

namespace DSO {

constexpr double TIME_EPS = 1e-12;

inline int64_t find_time_index(const std::vector<double>& grid, double t) {
    auto it = std::lower_bound(grid.begin(), grid.end(), t - TIME_EPS);
    if (it == grid.end()) return -1;
    if (std::abs(*it - t) > TIME_EPS) return -1;
    return (int64_t)std::distance(grid.begin(), it);
}

struct ControlIntervals {
    std::vector<double> start_times;
    std::vector<double> end_times;

    size_t n_intervals() const { return start_times.size(); }

    void validate(double eps = 1e-12) const {
        TORCH_CHECK(start_times.size() == end_times.size(), "start/end size mismatch");
        TORCH_CHECK(!start_times.empty(), "no control intervals");

        for (size_t k = 0; k < start_times.size(); ++k) {
            TORCH_CHECK(start_times[k] + eps < end_times[k], "non-positive interval");
            if (k + 1 < start_times.size()) {
                TORCH_CHECK(end_times[k] <= start_times[k + 1] + eps, "overlapping / non-causal intervals");
                TORCH_CHECK(start_times[k] <= start_times[k + 1] + eps, "start not nondecreasing");
                TORCH_CHECK(end_times[k] <= end_times[k + 1] + eps, "end not nondecreasing");
            }
        }
    }
};

struct BoundControlIntervals {
    std::vector<int64_t> start_idx;
    std::vector<int64_t> end_idx;
};

BoundControlIntervals bind_to_grid(const ControlIntervals& control_intervals, const std::vector<double>& grid) {
    BoundControlIntervals out;
    out.start_idx.reserve(control_intervals.n_intervals());
    out.end_idx.reserve(control_intervals.n_intervals());

    for (size_t k = 0; k < control_intervals.n_intervals(); ++k) {
        auto s = find_time_index(grid, control_intervals.start_times[k]);
        auto e = find_time_index(grid, control_intervals.end_times[k]);
        TORCH_CHECK(s >= 0 && e >= 0, "interval endpoints not on grid");
        TORCH_CHECK(s < e, "start index must be < end index");
        out.start_idx.push_back(s);
        out.end_idx.push_back(e);
    }
    return out;
}


inline std::vector<double> make_time_grid(double maturity, double dt, bool include_maturity = true) {
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
