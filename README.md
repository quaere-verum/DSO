# Differentiable Stochastic Optimisation (DSO)

High-performance **C++ framework for differentiable Monte Carlo and stochastic control**, built on top of **libtorch (PyTorch C++ API)**.

The project was developed as a research platform for experiments in deep hedging and risk-sensitive optimisation under stochastic volatility and transaction costs.

It demonstrates how **automatic differentiation can be combined with Monte Carlo simulation** to optimise hedging strategies directly from simulated market dynamics.

---

## Overview

Many financial optimisation problems can be written as:

$$
\min_{\theta}  \rho(PnL(\theta))
$$

where

* $\theta$ parameterises a hedging strategy or policy
* $PnL(\theta)$ is the resulting profit and loss distribution
* $\rho$ is a risk measure (e.g. variance, CVaR)

This project implements a framework where:

1. Market dynamics are simulated via Monte Carlo.
2. A parametric hedging policy interacts with the simulated market.
3. A risk objective is evaluated on the resulting PnL.
4. **Automatic differentiation computes gradients through the entire pipeline.**
5. Optimisers update the policy parameters (as well as other parameters, as needed).

The result is a **differentiable stochastic optimisation loop** capable of learning hedging strategies directly from simulated market models.

---

## Key Features

### Differentiable Monte Carlo

* Full simulation → hedging → risk evaluation pipeline is differentiable
* Gradients computed using **libtorch autograd**
* Supports optimisation of hedging policies with respect to risk objectives

---

### High-Performance Simulation

* Multi-threaded Monte Carlo using **Intel TBB**
* Efficient batched simulation
* CPU and CUDA execution supported through libtorch

---

### Modular Architecture

The framework separates core components:

```
StochasticModel
      ↓
MonteCarloExecutor
      ↓
HedgingEngine
      ↓
RiskMeasure
      ↓
Autograd Gradient
      ↓
Optimiser
```

This makes it easy to experiment with different:

* market models
* financial products
* hedging policies
* risk objectives

---

### Implemented Components

#### Market Models

* Black-Scholes model
* **Heston stochastic volatility model**
* Differentiable parameterisation for calibration or optimisation

---

#### Financial Products

Example implementations include:

* European options
* Asian options
* Lookback options
* Barrier options

---

#### Hedging Engine

Simulates discrete-time hedging with:

* transaction costs
* dynamic hedge updates
* path-dependent state tracking

---

#### Risk Objectives

Examples include:

* Mean Squared Error (variance-style objective)
* Mean-Variance
* Entropic Risk
* Expected Shortfall (CVaR)

CVaR is implemented using the Rockafellar–Uryasev formulation:

$$
\text{CVaR}_\alpha = z + \frac{1}{1-\alpha} \mathbb{E}[(L - z)^+]
$$

---

#### Controllers

Hedging policies (and feature extractors) are implemented as **PyTorch neural networks**, allowing:

* linear policies
* shallow neural networks
* recurrent latent state representations

---

## Example Research Experiment

The framework was used to train neural hedging policies for a **down-and-out call option under Heston dynamics with transaction costs** (see `scripts/run_barrier.ps1`).

Two optimisation objectives were compared:

* Mean Squared Error (variance minimisation)
* Conditional Value-at-Risk (tail-risk minimisation)

### Results

| Objective | CVaR(95) | Std(PnL) | Mean Turnover |
| --------- | -------- | -------- | ------------- |
| MSE       | 7.78     | 2.77     | 3.89          |
| CVaR      | **6.49** | 3.37     | **2.44**      |

Key observations:

* **CVaR optimisation reduces tail risk**
* CVaR policies **trade less frequently**, reducing transaction costs
* Different risk objectives lead to **qualitatively different hedging strategies**

---

## Technology Stack

* **C++23**
* **libtorch (PyTorch C++ API)**
* **Intel TBB** for parallel Monte Carlo
* CUDA support via PyTorch

---

## Motivation

This project was created to explore research questions such as:

* How do different risk objectives affect optimal hedging strategies?
* Can latent neural representations recover hidden volatility states?
* How does transaction cost structure interact with stochastic volatility?

The framework enables large-scale Monte Carlo experiments with **differentiable objectives**, making it possible to study these questions empirically.

---

## Example Usage

Pseudo-workflow:

```cpp
auto model = HestonModel(config);
auto controller = NeuralController(...);
auto objective = MCHedgeObjective(...);
auto optimiser = Adam(...);

MonteCarloGradientTrainer trainer(...);

for (int epoch = 0; epoch < N; ++epoch) {
    auto loss = optimiser.step(trainer);
}
```

The optimiser updates the hedging policy using gradients computed through the simulated market dynamics.

---

## Status

This repository represents **research code developed for experimentation**.

It was designed primarily as a personal research platform rather than a fully polished production framework.

---

## License

MIT License.