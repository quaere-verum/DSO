# DSO - Differentiable Stochastic Optimisation
## Overview
This repository contains a lightweight library for differentiable stochastic optimisation, with a particular focus on quantitative finance. The entire framework revolves around Monte Carlo simulation and PyTorch's autograd (using the C++ back-end libtorch). We use Intel's TBB to parallellise the resulting workloads efficiently. Examples of problems that can be solved in this framework are:
- Training a policy using policy gradient methods to perform stochastic optimal control (e.g. optimal hedging of a derivative)
- Calibration of market models w.r.t. market prices of instruments
- Computation of Greeks (first and second order) of custom derivative products w.r.t. market model parameters, for any user-defined market model.

See `main.cpp` for examples. The library is built to be modular and performant.

## Next Steps
- Implement a `TermStructure` class to incorporate discounting and interest rate models.
- Add more sophisticated market models, e.g. a `HestonModel`.
- Create a Python wrapper to allow `Product`'s to be defined in Python, then evaluated using the C++ engine.