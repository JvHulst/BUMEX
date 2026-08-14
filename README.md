# BUMEX: Bounded Uncertainty Model-based Exploration

This repository contains the implementation of **BUMEX** (Bounded Uncertainty Model-based Exploration), a reinforcement learning method that uses prior model knowledge to guide exploration and accelerate learning.

**📄 Paper:** [Smart Exploration in Reinforcement Learning Using Bounded Uncertainty Models](https://doi.org/10.1109/CDC57313.2025.11313018)

Presented at the 2025 IEEE 64th Conference on Decision and Control (CDC), Rio de Janeiro, Brazil

## Overview

BUMEX leverages bounded uncertainty models to compute Q-function bounds via convex optimization. These bounds can then be used to guide the RL agent's exploration in a clever way. We provide implementations of BUMEX alongside baseline exploration strategies for comparison.

**Implemented exploration policies:**
- **BUMEX (Exploring Policy)**: Our novel bounded uncertainty-based exploration 
- **Epsilon-Greedy**: Standard epsilon-greedy exploration
- **UCB1**: Upper Confidence Bound algorithm  
- **UCRL2**: Optimistic reinforcement learning
- **Thompson Sampling (PSRL)**: Posterior sampling for RL

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib gymnasium scipy

# Run single experiments
python experiments/frozen_lake.py
python experiments/cartpole.py  
python experiments/taxi.py

# Run statistical comparisons with Monte Carlo
python experiments/monte_carlo.py

# Analyze results in Jupyter notebook
jupyter notebook notebooks/results_comparison.ipynb
```

## Repository Structure

```
├── experiments/               # Experiment runners
├── src/                       # Core implementations
│   ├── exploring_policy.py    # BUMEX implementation
│   ├── fast_regularized.py    # Solver for the regularized Q-bound problem
│   ├── epsilon_greedy.py      # Baseline policies
│   ├── ucb1.py
│   ├── ucrl2.py
│   ├── thompson_sampling.py
│   ├── *_wrapper.py           # Environment interfaces
│   └── utils.py               # Utilities
├── config/                    # JSON configuration files  
├── tests/                     # Solver checks
└── notebooks/                 # Visualization of results
```

## Environments

**FrozenLake**: Discrete grid world with stochastic transitions
- Simple benchmark for tabular RL methods
- Configurable grid size and slip probability
- BUMEX uses model set that models adjacency in grid world as well as the exact reward function

**CartPole**: Pole balancing with continuous states  
- Finite state abstraction wrapper integrated to reduce to tabular RL
- Physics-based samples used to generate uncertainty model set
- Model bound caching system for computational speedup
- BUMEX uses an uncertain transition model and the exact reward function

**Taxi**: Discrete pickup/delivery task
- Slightly more complicated grid-world benchmark for tabular RL methods
- BUMEX uses model set that models adjacency in grid world and has an uncertain reward model

## Q-bound solver

Each Q-bound update optimizes, for every state-action pair, a transition distribution over a box and the simplex with a KL penalty towards the empirical distribution. This inner problem is separable with a single coupling constraint, so it admits a closed-form dual reduction; a general conic solver is therefore unnecessary, and exploiting the structure is worth roughly two orders of magnitude. `src/fast_regularized.py` is a purpose-built solver for it, using a closed-form fill where the regularization vanishes and a bisection on the dual multiplier elsewhere, and solving all state-action pairs of a sweep as one vectorized batch.

The problem is an instance of singly-constrained separable convex resource allocation, surveyed in Patriksson, *A survey on the continuous nonlinear resource allocation problem*, European Journal of Operational Research 185(1), 2008.

`python tests/test_fast_regularized.py` checks the solution against SciPy, against the closed form at zero regularization, and against the KKT conditions.

## Dependencies

- `numpy`: Numerical computations
- `gymnasium`: RL environments
- `scipy`: Spatial computations
- `matplotlib`: Visualization

## Citation

**If you use this code, please cite our paper:**

```bibtex
@inproceedings{vanHulst2025,
title = {Smart Exploration in Reinforcement Learning Using Bounded Uncertainty Models},
author = {van Hulst, J. S. and Heemels, W. P. M. H. and Antunes, D. J.},
booktitle = {2025 IEEE 64th Conference on Decision and Control (CDC)},
doi = {10.1109/CDC57313.2025.11313018},
eprint = {arxiv.org/abs/2504.05978},
month = {dec},
pages = {5132--5138},
publisher = {IEEE},
url = {https://ieeexplore.ieee.org/document/11313018/},
year = {2025}
}
```

**ArXiv version:** https://arxiv.org/abs/2504.05978
