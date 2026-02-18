#pragma once
#include <chrono>
#include <cstdint>
#include <array>
#include <string>
#include <iostream>

namespace DSO {

enum class Stage : int {
    Simulate = 0,
    Rng      = 1,
    Loss     = 2,
    Grad     = 3,
    Accum    = 4,
    SimMath  = 5,
    Count    = 6,
};

struct PerfCounters {
    std::array<uint64_t, static_cast<int>(Stage::Count)> ns{};
    std::array<uint64_t, static_cast<int>(Stage::Count)> calls{};
    void add(Stage s, uint64_t dt_ns) {
        auto i = static_cast<int>(s);
        ns[i] += dt_ns;
        calls[i] += 1;
    }
    void merge(const PerfCounters& o) {
        for (int i = 0; i < static_cast<int>(Stage::Count); ++i) {
            ns[i] += o.ns[i];
            calls[i] += o.calls[i];
        }
    }
};

struct ScopedTimer {
    PerfCounters& c;
    Stage s;
    std::chrono::high_resolution_clock::time_point t0;
    ScopedTimer(PerfCounters& c_, Stage s_)
        : c(c_), s(s_), t0(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer() {
        auto t1 = std::chrono::high_resolution_clock::now();
        uint64_t dt = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
        c.add(s, dt);
    }
};

inline const char* stage_name(Stage s) {
    switch (s) {
        case Stage::Simulate: return "simulate";
        case Stage::Rng:      return "rng";
        case Stage::Loss:     return "loss";
        case Stage::Grad:     return "autograd_grad";
        case Stage::Accum:    return "accumulate";
        case Stage::SimMath:  return "sim_math";
        default:              return "unknown";
    }
}

inline void print_perf(const PerfCounters& p, uint64_t total_paths) {
    uint64_t total_ns = 0;
    for (int i = 0; i < static_cast<int>(Stage::Count); ++i) total_ns += p.ns[i];

    std::cout << "\n=== Perf breakdown ===\n";
    std::cout << "total_paths=" << total_paths
              << " total_ms=" << (double)total_ns / 1e6
              << " ns/path=" << (double)total_ns / (double)total_paths
              << "\n";

    for (int i = 0; i < static_cast<int>(Stage::Count); ++i) {
        auto s = static_cast<Stage>(i);
        if (p.calls[i] == 0) continue;
        double ms = (double)p.ns[i] / 1e6;
        double pct = total_ns ? (100.0 * (double)p.ns[i] / (double)total_ns) : 0.0;
        std::cout << stage_name(s)
                  << ": " << ms << " ms (" << pct << "%)"
                  << " calls=" << p.calls[i]
                  << "\n";
    }
}

} // namespace DSO
