"""
Taxi experiment runner for BUMEX.

This module runs experiments on the Taxi-v3 environment.
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
from utils import create_run_dir, run_evaluation_episodes, initialize_policy, run_training_loop, save_experiment_results
from plotting import plot_evaluation_performance, plot_greedy_policy, plot_Q_bounds_evolution, plot_Q_bounds_snapshot, plot_thompson_sampling_models, visual_run


def main():
    """
    Run Taxi experiment with specified policy.
    """
    # Load configuration
    config = load_config('taxi')
    
    # Create results folder
    script_dir = os.path.dirname(__file__)
    results_folder = os.path.join(script_dir, "..", "results")
    os.makedirs(results_folder, exist_ok=True)
    results_subfolder = os.path.join(results_folder, "results_taxi")
    os.makedirs(results_subfolder, exist_ok=True)
    
    # Initialize environment
    env = gymnasium.make('Taxi-v3')
    
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
        visits=results["visits"],
        Q_storage=results["Q_storage"],
        Q_lower_storage=results["Q_lower_storage"] if results["Q_lower_storage"] else None,
        Q_upper_storage=results["Q_upper_storage"] if results["Q_upper_storage"] else None
    )
    
    # Plot evaluation performance
    eval_plot_path = os.path.join(run_dir, "evaluation_performance.png")
    plot_evaluation_performance(results["evaluation_records"], eval_plot_path)

    # Generate additional plots if verbose_plotting is enabled
    if config.get("verbose_plotting", False):
        print("Generating detailed visualizations...")
        
        # Plot greedy policy
        #TODO: Implement taxi-specific greedy policy visualization
        # May need custom plotting for 4D state space visualization
        
        # Plot Q-bounds evolution for exploring policy
        if policy_choice == "exploring_policy" and results["Q_lower_storage"]:
            # Add final Q-bounds values to storage for complete evolution plot
            Q_lower_final = results["Q_lower_storage"] + [policy.Q_lower.copy()]
            Q_upper_final = results["Q_upper_storage"] + [policy.Q_upper.copy()]
            bounds_plot_path = os.path.join(run_dir, "Q_bounds_evolution.png")
            plot_Q_bounds_evolution(Q_lower_final, Q_upper_final, results["Q_storage"], 
                                   num_episodes_before_policy_update,
                                   num_episodes_before_evaluation, env, bounds_plot_path,
                                   num_episodes, env_wrapper=policy.env_wrapper)
            
            # Plot final Q-bounds snapshot
            final_bounds_plot_path = os.path.join(run_dir, "Q_bounds_final.png")
            plot_Q_bounds_snapshot(policy.Q_lower, policy.Q_upper, policy.Q, env,
                                   env_wrapper=policy.env_wrapper,
                                   save_path=final_bounds_plot_path)

    print(f"Experiment completed. Results saved to: {run_dir}")


if __name__ == "__main__":
    main()
