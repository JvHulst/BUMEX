"""
CartPole experiment runner for BUMEX.

This module runs experiments on the CartPole environment using finite state abstraction.
"""

import os
import sys
import numpy as np
import gymnasium
import time
import json
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from config_utils import load_config
from finite_abstraction_wrapper import FiniteAbstractionWrapper
from utils import create_run_dir, initialize_policy, run_training_loop, save_experiment_results, generate_experiment_plots


def main():
    """
    Run CartPole experiment with specified policy using finite state abstraction.
    """
    # Load configuration
    config = load_config('cartpole')
    
    # Create results folder
    script_dir = os.path.dirname(__file__)
    results_folder = os.path.join(script_dir, "..", "results")
    os.makedirs(results_folder, exist_ok=True)
    results_subfolder = os.path.join(results_folder, "results_cartpole")
    os.makedirs(results_subfolder, exist_ok=True)
    
    # Initialize environment with finite abstraction wrapper
    base_env = gymnasium.make('CartPole-v1')
    env = FiniteAbstractionWrapper(base_env, config["discretization"])
    
    # Get parameters from config
    alpha = config["alpha"]
    gamma = config["gamma"]
    num_episodes = config["num_episodes"]
    num_episodes_before_evaluation = config["num_episodes_before_evaluation"]
    num_episodes_before_policy_update = config["num_episodes_before_policy_update"]
    policy_choice = config["policy_choice"]
    
    # Initialize policy using shared function
    policy = initialize_policy(policy_choice, env, config)
    
    # Create run directory
    run_dir = create_run_dir(results_subfolder, policy_choice, num_episodes)
    
    # Run training loop using shared function
    results = run_training_loop(env, policy, config, run_dir)
    
    # Save results using shared function
    save_experiment_results(
        run_dir, 
        config, 
        results["episode_rewards"], 
        results["evaluation_records"],
        results["visits"],
        results["Q_storage"],
        results["Q_lower_storage"],
        results["Q_upper_storage"]
    )
    
    # Generate plots
    generate_experiment_plots(env, policy, results, config, run_dir)
    
    print(f"Experiment completed. Results saved to {run_dir}")


if __name__ == "__main__":
    main()
