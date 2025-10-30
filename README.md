# BUMEX: Bounded Uncertainty Model-based Exploration

This repository contains the implementation of **BUMEX** (Bounded Uncertainty Model-based Exploration), a reinforcement learning method that uses prior model knowledge to guide exploration and accelerate learning.

**📄 Paper:** [Smart Exploration in Reinforcement Learning Using Bounded Uncertainty Models](https://arxiv.org/abs/2504.05978) (2025 64th IEEE Conference on Decision and Control (CDC), Rio de Janeiro, Brazil)

## Overview

BUMEX leverages bounded uncertainty models to compute Q-function bounds via convex optimization, enabling smarter exploration compared to standard methods. We provide implementations of BUMEX alongside baseline exploration strategies for comparison.

**Implemented exploration policies:**
- **BUMEX (Exploring Policy)**: Our novel bounded uncertainty-based exploration 
- **Epsilon-Greedy**: Standard ε-greedy exploration
- **UCB1**: Upper Confidence Bound algorithm  
- **UCRL2**: Optimistic reinforcement learning
- **Thompson Sampling (PSRL)**: Posterior sampling for RL

## Quick Start

```bash
# Install dependencies
pip install numpy matplotlib cvxpy gymnasium scipy mosek

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
├── src/                        # Core implementations
│   ├── exploring_policy.py     # BUMEX implementation
│   ├── epsilon_greedy.py       # Baseline policies
│   ├── ucb1.py
│   ├── ucrl2.py
│   ├── thompson_sampling.py
│   ├── *_wrapper.py           # Environment interfaces
│   └── utils.py               # Utilities
├── experiments/               # Experiment runners
├── config/                   # JSON configuration files  
└── notebooks/               # Analysis tools
```

## Environments

**FrozenLake**: Discrete grid world with stochastic transitions
- Standard benchmark for tabular RL methods
- Configurable grid size and slip probability

**CartPole**: Pole balancing with continuous states  
- Finite state abstraction for discrete RL algorithms
- Physics-based uncertainty modeling
- Efficient caching system for computational speedup

**Taxi**: Discrete pickup/delivery task
- Multi-objective navigation problem
- Hierarchical state structure

## Dependencies

- `numpy`: Numerical computations
- `cvxpy`: Convex optimization for Q-bounds computation  
- `mosek`: High-performance optimization solver (recommended for best performance)
- `gymnasium`: RL environments
- `scipy`: Spatial computations
- `matplotlib`: Visualization

**Note:** While other solvers can be used with CVXPY, MOSEK provides significantly better performance for the convex optimization problems in BUMEX.

## Current Status

The codebase is functional with key features implemented. Ongoing consolidation includes:
- Code cleanup and documentation improvements
- Performance optimizations
- Enhanced error handling

## Citation

**If you use this code, please cite our paper:**

```bibtex
@inproceedings{hulst2025,
  title={Smart Exploration in Reinforcement Learning Using Bounded Uncertainty Models},
  archivePrefix = {arXiv},
  arxivId = {2504.05978},
  eprint = {2504.05978},
  author={van Hulst, J. S. and Heemels, W P M H and Antunes, D J},
  booktitle={2025 64th IEEE Conference on Decision and Control (CDC)},
  publisher = {IEEE},
  url = {https://arxiv.org/abs/2504.05978},
  year={2025}
}
```

**ArXiv preprint:** https://arxiv.org/abs/2504.05978
