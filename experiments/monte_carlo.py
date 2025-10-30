"""
Monte Carlo experiment runner for BUMEX.

This module orchestrates multiple runs of experiments for statistical analysis.
Adapted from monte_carlo_runs.ipynb to work with the new modular framework.
"""

import os
import sys
import subprocess
import time
import shutil
import json
import numpy as np

# Add src to path for plotting functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from plotting import print_single_run_summary, plot_monte_carlo_results


def run_monte_carlo_experiments(environment='frozen_lake', num_runs=100, config_override=None):
    """
    Run Monte Carlo experiments for specified environment.
    
    Args:
        environment (str): Environment name ('frozen_lake', 'cartpole', etc.)
        num_runs (int): Number of Monte Carlo runs
        config_override (dict): Optional config parameters to override
    """
    # Determine experiment script and results directory based on environment
    if environment == 'frozen_lake':
        experiment_script = 'frozen_lake.py'
        base_results_dir = 'results_frozen_lake'
    elif environment == 'cartpole':
        experiment_script = 'cartpole.py'  # To be implemented
        base_results_dir = 'results_cartpole'
    elif environment == 'taxi':
        experiment_script = 'taxi.py'
        base_results_dir = 'results_taxi'
    else:
        raise ValueError(f"Environment {environment} not supported yet")
    
    # Get script directory and set up paths
    script_dir = os.path.dirname(__file__)
    experiment_path = os.path.join(script_dir, experiment_script)
    results_path = os.path.join(script_dir, "..", "results", base_results_dir)
    
    # Create results directory
    os.makedirs(results_path, exist_ok=True)
    
    # Load default configuration
    config_path = os.path.join(script_dir, "..", "config", f"{environment}.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Apply config overrides if provided
    if config_override:
        config.update(config_override)
    
    # Extract key parameters for folder naming
    policy_choice = config.get("policy_choice", "unknown")
    num_episodes = config.get("num_episodes", 0)
    
    # Create timestamped Monte Carlo results folder
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    monte_carlo_folder = os.path.join(
        results_path, 
        f"monte_carlo_{num_runs}_runs_{policy_choice}_{num_episodes}_{timestamp}"
    )
    os.makedirs(monte_carlo_folder, exist_ok=True)
    
    # Save configuration to Monte Carlo folder
    settings_path = os.path.join(monte_carlo_folder, "settings.json")
    with open(settings_path, "w") as f:
        json.dump(config, f, indent=4)
    
    print(f"Created Monte Carlo results folder: {monte_carlo_folder}")
    print(f"\n\n Running {num_runs} experiments on {environment} with {policy_choice} policy...\n\n")
    
    evaluation_returns = []
    
    # Run Monte Carlo experiments
    for i in range(1, num_runs + 1):
        print(f"Starting Monte Carlo run {i} out of {num_runs}...")
        
        # Execute experiment script with override config
        env = os.environ.copy()
        env['BUMEX_MONTE_CARLO_CONFIG'] = settings_path
        env['MPLBACKEND'] = 'Agg'  # Use non-interactive backend to prevent figure display
        result = subprocess.run(["python", experiment_path], cwd=script_dir, capture_output=True, text=True, env=env)
        
        if result.returncode != 0:
            print(f"Error in run {i}: {result.stderr}")
            continue
        
        time.sleep(1)  # Wait to ensure filesystem timestamps are updated
        
        # Find the most recent run folder created by the experiment script
        run_dirs = [d for d in os.listdir(results_path) 
                   if os.path.isdir(os.path.join(results_path, d)) and d != os.path.basename(monte_carlo_folder)]
        
        if not run_dirs:
            print(f"No results folder found in {results_path}; skipping move.")
            continue
        
        latest_run = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(results_path, d)))
        src = os.path.join(results_path, latest_run)
        
        # Create destination folder with run index
        dst = os.path.join(monte_carlo_folder, f"{i:0{len(str(num_runs))}d}_{latest_run}")
        shutil.move(src, dst)
        
        # Load evaluation records
        eval_file = os.path.join(dst, "evaluation_records.npy")
        if os.path.exists(eval_file):
            evaluation_records = np.load(eval_file)
            evaluation_returns.append(evaluation_records)
            # Print compact summary of this run
            print_single_run_summary(evaluation_records, i)
        else:
            print(f"Warning: No evaluation_records.npy found in run {i}")
    
    print("All Monte Carlo runs have been completed.")
    
    # Process and visualize results if we have data
    if evaluation_returns:
        # Create and show final plot
        plot_path = os.path.join(monte_carlo_folder, "evaluation_returns.png")
        plot_monte_carlo_results(evaluation_returns, policy_choice, plot_path, show_plot=True)
        
        print(f"Results saved to: {monte_carlo_folder}")
        print(f"Plot saved to: {plot_path}")
    else:
        print("No evaluation data found to process.")
    
    return monte_carlo_folder


if __name__ == "__main__":
    # Example usage: run Monte Carlo experiments

    import time

    run_monte_carlo_experiments(
        environment='cartpole',
        num_runs=22,
        config_override={"verbose_plotting": False, 'num_episodes': 10000,
                         "policy_choice": "exploring_policy", "num_episodes_before_policy_update": 250,
                         "c": 5.0}
    )