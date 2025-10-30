"""
General utilities for BUMEX.

This module contains common utility functions used across experiments.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt


def load_evaluation_records(folder):
    """
    Load evaluation records from a results folder.
    
    Args:
        folder (str): Path to results folder
        
    Returns:
        np.ndarray: 2D array with columns [episode, avg_return], or None if not found
    """
    file_path = os.path.join(folder, 'evaluation_records.npy')
    if os.path.exists(file_path):
        return np.load(file_path, allow_pickle=True)
    else:
        print(f'No evaluation_records.npy found in {folder}')
        return None


def load_monte_carlo_evaluations(mc_folder):
    """
    Load evaluation records from all subfolders in a Monte Carlo results folder.
    
    Args:
        mc_folder (str): Path to Monte Carlo results folder
        
    Returns:
        list: List of evaluation record arrays, or None if none found
    """
    subfolders = [os.path.join(mc_folder, d) for d in os.listdir(mc_folder) 
                  if os.path.isdir(os.path.join(mc_folder, d))]
    records_list = []
    for sub in subfolders:
        rec = load_evaluation_records(sub)
        if rec is not None:
            records_list.append(rec)
    return records_list if records_list else None


def create_run_dir(base_folder, policy_choice, num_episodes, map_type=None):
    """
    Create a timestamped directory for experiment results.
    
    Args:
        base_folder (str): Base results directory
        policy_choice (str): Name of the policy being used
        num_episodes (int): Number of episodes in the experiment
        map_type (str, optional): Environment map type
        
    Returns:
        str: Path to created directory
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    if map_type:
        run_dir = os.path.join(base_folder, f"run_{policy_choice}_episodes_{num_episodes}_{map_type}_{timestamp}")
    else:
        run_dir = os.path.join(base_folder, f"run_{policy_choice}_episodes_{num_episodes}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_evaluation_episodes(env, policy, num_eval_episodes=100):
    """
    Run evaluation episodes using greedy policy.
    
    Args:
        env: OpenAI Gym environment
        policy: Policy object with get_greedy_action method
        num_eval_episodes (int): Number of evaluation episodes
        
    Returns:
        float: Average return over evaluation episodes
    """
    total_reward = 0
    for _ in range(num_eval_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False
        while not (done or truncated):
            action = policy.get_greedy_action(state)  # Use policy's greedy action for evaluation
            state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_eval_episodes


def initialize_policy(policy_choice, env, config):
    """
    Initialize the specified policy with configuration parameters.
    
    Args:
        policy_choice (str): Name of the policy to initialize
        env: OpenAI Gym environment (may be wrapped with FiniteAbstractionWrapper)
        config (dict): Configuration dictionary containing policy parameters
        
    Returns:
        Policy object
    """
    # Import here to avoid circular imports
    from epsilon_greedy import EpsilonGreedyPolicy
    from ucb1 import UCB1Policy
    from ucrl2 import UCRL2Policy
    from thompson_sampling import ThompsonSamplingPolicy
    from exploring_policy import ExploringPolicy
    
    # Create environment wrapper if needed based on policy requirements
    env_wrapper = None
    
    # Check if policy requires environment wrapper
    wrapper_required_policies = ["thompson_sampling", "exploring_policy"]
    if policy_choice in wrapper_required_policies:
        # Detect environment type and create appropriate wrapper
        # Check environment ID first (most reliable)
        env_id = str(env.unwrapped.spec.id) if hasattr(env.unwrapped, 'spec') else 'Unknown'
        
        if 'Taxi' in env_id:  # Taxi environment
            from taxi_wrapper import TaxiWrapper
            env_wrapper = TaxiWrapper(env)
        elif 'CartPole' in env_id or hasattr(env, 'discretize_state'):  # CartPole environment
            from cartpole_wrapper import CartPoleWrapper
            env_wrapper = CartPoleWrapper(env, config)
        elif hasattr(env.unwrapped, 'desc') and ('FrozenLake' in env_id or 'Lake' in env_id):  # FrozenLake environment
            from frozen_lake_wrapper import FrozenLakeWrapper
            env_wrapper = FrozenLakeWrapper(env)
        else:
            # Unknown environment - raise error for wrapper-required policies
            raise ValueError(f"Policy '{policy_choice}' requires environment wrapper, but no wrapper available for environment '{env_id}'. "
                           f"Supported environments: FrozenLake, CartPole, Taxi")
    
    alpha = config["alpha"]
    gamma = config["gamma"]
    num_episodes = config["num_episodes"]
    
    if policy_choice == "epsilon_greedy":
        return EpsilonGreedyPolicy(
            env,
            config["epsilon_initial"],
            config["epsilon_final"],
            num_episodes,
            gamma=gamma,
            alpha=alpha,
            alpha_final=config.get("alpha_final", alpha)
        )
    elif policy_choice == "ucb1":
        return UCB1Policy(
            env,
            config.get("c", 2.0),
            num_episodes,
            gamma=gamma,
            alpha=alpha,
            alpha_final=config.get("alpha_final", alpha)
        )
    elif policy_choice == "ucrl2":
        return UCRL2Policy(
            env,
            config.get("c", 2.0),
            config.get("delta", 0.1),
            config["epsilon_initial"],
            config["epsilon_final"],
            num_episodes,
            gamma=gamma,
        )
    elif policy_choice == "thompson_sampling":
        if not env_wrapper:
            raise ValueError(f"Environment wrapper is required for thompson_sampling policy")
        return ThompsonSamplingPolicy(env,
            env_wrapper=env_wrapper,
            gamma=gamma,
            use_probabilistic_rewards=config.get("use_probabilistic_rewards", False),
        )
    elif policy_choice == "exploring_policy":
        if not env_wrapper:
            raise ValueError(f"Environment wrapper is required for exploring_policy policy")
        return ExploringPolicy(
            env,
            env_wrapper=env_wrapper,
            c=config.get("c", 2.0),
            delta=config.get("delta", 0.1),
            epsilon_initial=config["epsilon_initial"],
            epsilon_final=config["epsilon_final"],
            num_episodes=num_episodes,
            gamma=gamma,
            alpha=alpha,
            alpha_final=config.get("alpha_final", alpha)
        )
    else:
        raise ValueError(f"Policy {policy_choice} not implemented")


def update_policy(policy, policy_choice, episode, config):
    """
    Apply policy-specific updates during training.
    
    Args:
        policy: Policy object
        policy_choice (str): Name of the policy
        episode (int): Current episode number
        config (dict): Configuration parameters
    """
    # Always update epsilon for policies that have it
    if hasattr(policy, 'update_epsilon'):
        policy.update_epsilon()

    if hasattr(policy, 'update_alpha'):
        policy.update_alpha()

    # Policy-specific periodic updates
    update_interval = config.get("num_episodes_before_policy_update", 100)
    
    if (episode + 1) % update_interval == 0:
        if policy_choice == "ucrl2":
            policy.update_optimistic_models()
        elif policy_choice == "thompson_sampling":
            policy.resample_policy()
        elif policy_choice == "exploring_policy":
            if 'CartPole' in str(policy.env.unwrapped.spec.id):
                # For CartPole, update Q-bounds less frequently due to slower learning
                if episode < 6000 and (
                    (episode >= 3000 and (episode + 1) % (2 * update_interval) == 0) or 
                    (episode >= 1500 and episode < 3000 and (episode + 1) % update_interval == 0) or 
                    (episode < 1500)
                ):
                    policy.update_Q_bounds(tol=1e-1)


def save_experiment_results(run_dir, config, episode_rewards, evaluation_records, 
                           visits=None, Q_storage=None, Q_lower_storage=None, Q_upper_storage=None):
    """
    Save experiment results to files.
    
    Args:
        run_dir (str): Directory to save results
        config (dict): Experiment configuration
        episode_rewards (list): Rewards per episode
        evaluation_records (list): Evaluation performance records
        visits (np.ndarray, optional): State visit counts
        Q_storage (list, optional): Q-table storage over time
        Q_lower_storage (list, optional): Lower Q-bounds storage
        Q_upper_storage (list, optional): Upper Q-bounds storage
    """
    # Save configuration
    with open(os.path.join(run_dir, "settings.json"), "w") as f:
        json.dump(config, f, indent=4)
    
    # Save basic results
    np.save(os.path.join(run_dir, "episode_rewards.npy"), np.array(episode_rewards))
    np.save(os.path.join(run_dir, "evaluation_records.npy"), np.array(evaluation_records))
    
    # Save optional arrays if provided
    if visits is not None:
        np.save(os.path.join(run_dir, "state_visits.npy"), visits)
    if Q_storage is not None:
        np.save(os.path.join(run_dir, "Q.npy"), np.array(Q_storage))
    if Q_lower_storage is not None:
        np.save(os.path.join(run_dir, "Q_lower_storage.npy"), np.array(Q_lower_storage))
    if Q_upper_storage is not None:
        np.save(os.path.join(run_dir, "Q_upper_storage.npy"), np.array(Q_upper_storage))


def run_training_loop(env, policy, config, run_dir):
    """
    Run the main training loop for any environment and policy.
    
    Args:
        env: OpenAI Gym environment
        policy: Policy object
        config (dict): Configuration parameters
        run_dir (str): Directory to save results
        
    Returns:
        dict: Training results including episode_rewards, evaluation_records, etc.
    """
    # Import here to avoid circular imports
    from plotting import plot_thompson_sampling_models
    from plotting import plot_Q_bounds_snapshot
    
    # Extract parameters
    num_episodes = config["num_episodes"]
    num_episodes_before_evaluation = config["num_episodes_before_evaluation"]
    policy_choice = config["policy_choice"]
    
    # Training storage
    episode_rewards = []
    visits = np.zeros(env.observation_space.n)
    evaluation_records = []
    Q_storage = [policy.Q.copy()]
    Q_lower_storage = []
    Q_upper_storage = []
    
    # Store initial Q-bounds for exploring policy
    if policy_choice == "exploring_policy":
        Q_lower_storage.append(policy.Q_lower.copy())
        Q_upper_storage.append(policy.Q_upper.copy())
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        terminated = False
        truncated = False

        # Episode termination uses (terminated or truncated) for proper gym compatibility
        while not (terminated or truncated):
            action = policy.choose_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            policy.update_q(state, action, reward, next_state, terminated, truncated)
            
            state = next_state
            episode_reward += reward
            visits[state] += 1
        
        episode_rewards.append(episode_reward)
        
        # Apply policy updates
        update_policy(policy, policy_choice, episode, config)
        
        # Evaluation
        if (episode) % num_episodes_before_evaluation == 0 or episode == num_episodes - 1:
            avg_return = run_evaluation_episodes(env, policy)
            evaluation_records.append([episode + 1, avg_return])
            Q_storage.append(policy.Q.copy())
            
            status_info = policy.get_status_info()
            status_str = ", ".join([f"{k} = {v}" for k, v in status_info.items()])
            print(f"Episode {episode + 1}: Average return = {avg_return:.3f}, {status_str}")
        
        # Handle special visualizations for Thompson Sampling
        # TODO: move this to update_policy
        update_interval = config.get("num_episodes_before_policy_update", 100)
        if ((episode + 1) % update_interval == 0 and policy_choice == "thompson_sampling" and 
            config.get("verbose_plotting", False)):
            thompson_plot_path = os.path.join(run_dir, f"thompson_models_episode_{episode+1}.png")
            plot_thompson_sampling_models(policy, env, thompson_plot_path)
        
        # Store Q-bounds for exploring policy
        # TODO: move this to update_policy
        if ((episode + 1) % update_interval == 0 and policy_choice == "exploring_policy"):
            Q_lower_storage.append(policy.Q_lower.copy())
            Q_upper_storage.append(policy.Q_upper.copy())
            if config.get("verbose_plotting", False):
                plot_Q_bounds_snapshot(policy.Q_lower, policy.Q_upper, policy.Q, env, env_wrapper=policy.env_wrapper)

    
    return {
        "episode_rewards": episode_rewards,
        "evaluation_records": evaluation_records,
        "visits": visits,
        "Q_storage": Q_storage,
        "Q_lower_storage": Q_lower_storage,
        "Q_upper_storage": Q_upper_storage
    }


def generate_experiment_plots(env, policy, results, config, run_dir):
    """
    Generate plots after experiment completion.
    
    Args:
        env: Environment object
        policy: Policy object  
        results: Results dictionary from run_training_loop
        config: Configuration dictionary
        run_dir: Directory to save plots
    """
    from plotting import plot_evaluation_performance, plot_greedy_policy, plot_cartpole_Q_evolution
    
    # Always plot evaluation performance
    performance_path = os.path.join(run_dir, "performance.png")
    plot_evaluation_performance(results["evaluation_records"], performance_path)
    
    # Plot greedy policy if verbose plotting enabled
    if config.get("verbose_plotting", False) and hasattr(policy, 'Q'):
        policy_path = os.path.join(run_dir, "policy.png")
        plot_greedy_policy(policy, env, policy_path)
        
        # CartPole-specific Q evolution plot
        if hasattr(env, 'state_bins') and results["Q_storage"]:
            eval_interval = config["num_episodes_before_evaluation"]
            q_evolution_path = os.path.join(run_dir, "Q_evolution_grid.png")
            plot_cartpole_Q_evolution(
                results["Q_storage"], env, eval_interval, q_evolution_path,
                results.get("Q_lower_storage"), results.get("Q_upper_storage"),
                config.get("num_episodes_before_policy_update")
            )


def load_q_bounds(folder):
    """
    Load Q-bounds data from folder.
    
    Args:
        folder (str): Path to results folder
        
    Returns:
        tuple: (Q, Q_lower_storage, Q_upper_storage) or (None, None, None) if not found
    """
    Q_path = os.path.join(folder, 'Q.npy')
    Q_lower_path = os.path.join(folder, 'Q_lower_storage.npy')
    Q_upper_path = os.path.join(folder, 'Q_upper_storage.npy')
    if os.path.exists(Q_path) and os.path.exists(Q_lower_path) and os.path.exists(Q_upper_path):
        Q = np.load(Q_path, allow_pickle=True)
        Q_lower_storage = np.load(Q_lower_path, allow_pickle=True)
        Q_upper_storage = np.load(Q_upper_path, allow_pickle=True)
        return Q, Q_lower_storage, Q_upper_storage
    else:
        print(f'Could not load Q-bounds from {folder}.')
        return None, None, None


def load_q_bounds_monte_carlo(folder):
    """
    Load Q-bounds from Monte Carlo folder (median across runs).
    
    Args:
        folder (str): Path to Monte Carlo results folder
        
    Returns:
        tuple: (Q_median, Q_lower_median, Q_upper_median)
    """
    subfolders = [os.path.join(folder, d) for d in os.listdir(folder) 
                  if os.path.isdir(os.path.join(folder, d))]
    Q_list, Q_lower_list, Q_upper_list = [], [], []
    for sub in subfolders:
        Q, Q_lower, Q_upper = load_q_bounds(sub)
        if Q is not None and Q_lower is not None and Q_upper is not None:
            Q_list.append(Q)
            Q_lower_list.append(Q_lower)
            Q_upper_list.append(Q_upper)
    Q_median = np.median(Q_list, axis=0)
    Q_lower_median = np.median(Q_lower_list, axis=0)
    Q_upper_median = np.median(Q_upper_list, axis=0)
    
    return Q_median, Q_lower_median, Q_upper_median


def load_q(folder):
    """
    Load Q function from folder.
    
    Args:
        folder (str): Path to results folder
        
    Returns:
        np.ndarray: Q function array, or None if not found
    """
    Q_path = os.path.join(folder, 'Q.npy')
    if os.path.exists(Q_path):
        return np.load(Q_path, allow_pickle=True)
    else:
        print(f'No Q.npy found in {folder}')
        return None
