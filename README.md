# DSO - Differentiable Stochastic Optimisation
## Overview
This repository contains a lightweight library for differentiable stochastic optimisation, with a particular focus on quantitative finance. The entire framework revolves around Monte Carlo simulation and PyTorch's autograd (using the C++ back-end libtorch). We use CUDA or Intel's TBB to parallellise the resulting workloads efficiently. Examples of problems that can be solved in this framework are:
- Training a policy using policy gradient methods to perform stochastic optimal control (e.g. optimal hedging of a derivative)
- Calibration of market models w.r.t. market prices of instruments
- Computation of Greeks (first and second order) of custom derivative products w.r.t. market model parameters, for any user-defined market model.

See `experiments` for examples. The library is built to be modular and performant.

## Next Steps
- Implement a `TermStructure` class to incorporate discounting and interest rate models.