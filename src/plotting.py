"""
Plotting and visualization utilities for BUMEX.

This module contains functions for generating plots and visualizations.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def plot_evaluation_performance(evaluation_records, save_path=None):
    """
    Plot evaluation performance over training episodes.
    
    Args:
        evaluation_records (list): List of (episode, avg_return) tuples
        save_path (str, optional): Path to save plot
    """
    episodes_eval, avg_returns = zip(*evaluation_records)
    plt.figure(figsize=(10, 5))
    plt.plot(episodes_eval, avg_returns, marker='o', linestyle='-', label='Greedy Evaluation Return')
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Greedy Return')
    plt.title('Greedy Evaluation Performance vs Training Episodes')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_greedy_policy(policy, env, save_path=None):
    """
    Plot greedy policy from policy object.
    
    Args:
        policy: Policy object with get_greedy_action method
        env: OpenAI Gym environment
        save_path (str, optional): Path to save the plot
    """
    # Check if this is a FrozenLake environment
    if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'desc'):
        _plot_greedy_policy_frozen_lake(policy, env, save_path)
    # Check if this is a CartPole environment with finite abstraction
    elif hasattr(env, 'state_bins') and hasattr(env, 'state_ranges'):
        _plot_greedy_policy_cartpole(policy, env, save_path)
    # Check if this is a Taxi environment
    elif hasattr(env, 'unwrapped') and 'Taxi' in str(env.unwrapped.spec.id):
        plot_taxi_greedy_policy(policy, env, save_path)
    else:
        print("Warning: plot_greedy_policy currently only supports FrozenLake, CartPole, and Taxi environments")
        pass


def _plot_greedy_policy_cartpole(policy, env, save_path=None):
    """Plot greedy policy for CartPole environment with finite abstraction."""
    if hasattr(policy, 'Q'):
        plot_cartpole_greedy_policy(policy, env, save_path)
    else:
        print("Warning: Policy does not have Q-function for visualization")


def _plot_greedy_policy_frozen_lake(policy, env, save_path=None):
    """Plot greedy policy for FrozenLake environment."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    grid_size = int(np.sqrt(n_states))
    
    action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    
    for s in range(n_states):
        i = s // grid_size
        j = s % grid_size
        ax = axs[i, j] if grid_size > 1 else axs
        
        action = policy.get_greedy_action(s)
        
        symbol = action_arrows.get(action, '?')
        ax.text(0, 0, symbol, ha='center', va='center', color='black',
                fontsize=20, fontweight='bold')
        ax.set_title(f'State {s}')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
    
    plt.suptitle('Greedy Policy')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_Q_bounds_evolution(Q_lower_storage, Q_upper_storage, Q_storage, 
                           num_episodes_before_policy_update,  
                           num_episodes_before_evaluation, env, save_path=None,
                           final_episode=None, env_wrapper=None):
    """
    Plot the evolution of Q-bounds over training.
    
    Args:
        Q_lower_storage (list): Stored Q-lower bounds
        Q_upper_storage (list): Stored Q-upper bounds
        Q_storage (list): Stored Q-values
        num_episodes_before_policy_update (int): Policy update frequency
        num_episodes_before_evaluation (int): Q-value evaluation frequency
        env: OpenAI Gym environment
        save_path (str, optional): Path to save plot
        final_episode (int, optional): Final episode number
        env_wrapper: Environment wrapper (for Taxi environment)
    """
    # Check if this is a FrozenLake environment
    if hasattr(env, 'unwrapped') and 'Frozen' in str(env.unwrapped.spec.id):
        _plot_Q_bounds_evolution_frozen_lake(Q_lower_storage, Q_upper_storage, Q_storage, 
                                            num_episodes_before_policy_update,
                                            num_episodes_before_evaluation, env, save_path,
                                            final_episode)
    # Check if this is a Taxi environment
    elif hasattr(env, 'unwrapped') and 'Taxi' in str(env.unwrapped.spec.id):
        plot_taxi_Q_bounds_evolution(Q_lower_storage, Q_upper_storage, Q_storage,
                                     num_episodes_before_policy_update, 
                                     num_episodes_before_evaluation, env_wrapper,
                                     save_path, final_episode)
    else:
        print("Warning: plot_Q_bounds_evolution currently only supports FrozenLake and Taxi environments")
        pass


def _plot_Q_bounds_evolution_frozen_lake(Q_lower_storage, Q_upper_storage, Q_storage, 
                                        num_episodes_before_policy_update,  
                                        num_episodes_before_evaluation, env, save_path=None,
                                        final_episode=None):
    """Plot Q-bounds evolution for FrozenLake environment."""
    if not Q_lower_storage or not Q_upper_storage:
        print("No Q-bounds data available for plotting")
        return
    
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    grid_size = int(np.sqrt(n_states))
    
    num_updates = len(Q_lower_storage)
    
    # Q-bounds updates at specified intervals, plus final episode if provided
    if final_episode is not None and num_updates > 0:
        # Regular updates plus final episode
        x_updates = list(np.arange(num_updates - 1) * num_episodes_before_policy_update) + [final_episode]
    else:
        x_updates = np.arange(num_updates) * num_episodes_before_policy_update
    
    # Q-values stored at evaluation intervals
    x_episodes = np.arange(len(Q_storage)) * num_episodes_before_evaluation
    
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    colors = plt.cm.tab10(np.linspace(0, 1, n_actions))
    
    for s in range(n_states):
        i = s // grid_size
        j = s % grid_size
        ax = axs[i, j] if grid_size > 1 else axs
        
        for a in range(n_actions):
            # Q-bounds evolution
            lower_vals = [Q_lower_storage[k][s, a] for k in range(num_updates)]
            upper_vals = [Q_upper_storage[k][s, a] for k in range(num_updates)]
            ax.plot(x_updates, lower_vals, color=colors[a], linestyle='--', marker='^', 
                   markersize=3, label=f'Action {a}' if s == 0 else "")
            ax.plot(x_updates, upper_vals, color=colors[a], linestyle='--', marker='v', markersize=3)
            
            # Q-values evolution
            q_vals = [Q_storage[k][s, a] for k in range(len(Q_storage))]
            ax.plot(x_episodes, q_vals, color=colors[a], linestyle='-', linewidth=1)
        
        ax.set_xlabel("Episodes")
        ax.set_ylabel("Q-value")
        if s == 0:
            ax.legend(fontsize='small')
        ax.set_title(f"State {s}")
    
    plt.suptitle("Q-bounds Evolution per State and Action")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_Q_bounds_snapshot(Q_lower, Q_upper, Q, env, env_wrapper=None, save_path=None):
    """
    Plot current Q-bounds and Q-values for all state-action pairs.
    
    Args:
        Q_lower (np.ndarray): Lower Q-bounds
        Q_upper (np.ndarray): Upper Q-bounds  
        Q (np.ndarray): Current Q-values
        env: OpenAI Gym environment
        save_path (str, optional): Path to save plot
        env_wrapper: Environment wrapper (for Taxi environment)
    """
    # Check if this is a FrozenLake environment
    if hasattr(env, 'unwrapped') and 'Frozen' in str(env.unwrapped.spec.id):
        _plot_Q_bounds_snapshot_frozen_lake(Q_lower, Q_upper, Q, env, save_path)
    elif hasattr(env, 'unwrapped') and 'Taxi' in str(env.unwrapped.spec.id):
        plot_taxi_Q_bounds_snapshot(Q_lower, Q_upper, Q, env_wrapper, save_path)
    else:
        print("Warning: plot_Q_bounds_snapshot currently only supports FrozenLake and Taxi environments")
        pass


def _plot_Q_bounds_snapshot_frozen_lake(Q_lower, Q_upper, Q, env, save_path=None):
    """Plot Q-bounds snapshot for FrozenLake environment."""
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    grid_size = int(np.sqrt(n_states))
    
    fig, axs = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    
    for s in range(n_states):
        i = s // grid_size
        j = s % grid_size
        ax = axs[i, j] if grid_size > 1 else axs
        
        x = np.arange(n_actions)
        width = 0.25
        
        ax.bar(x - width, Q_lower[s, :], width=width, label='Q_lower', color='C0')
        if Q is not None:
            ax.bar(x, Q[s, :], width=width, label='Q', color='C1')
        ax.bar(x + width, Q_upper[s, :], width=width, label='Q_upper', color='C2')
        
        ax.set_xticks(x)
        ax.set_xticklabels([str(a) for a in range(n_actions)])
        ax.set_title(f"State {s}")
        if s == 0:
            ax.legend(fontsize='small')
    
    plt.suptitle('Q-bounds per State')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_thompson_sampling_models(policy, env, save_path=None):
    """
    Visualize Thompson Sampling model heatmaps comparing empirical vs sampled models.
    
    Args:
        policy: Thompson sampling policy with model access
        env: FrozenLake environment
        save_path (str, optional): Path to save plot
    """
    # Check if this is a FrozenLake environment
    if hasattr(env, 'unwrapped') and 'Frozen' in str(env.unwrapped.spec.id):
        _plot_thompson_sampling_models_frozen_lake(policy, env, save_path)
    else:
        print("Warning: plot_thompson_sampling_models currently only supports FrozenLake environments")
        pass


def _plot_thompson_sampling_models_frozen_lake(policy, env, save_path=None):
    """Compare sampled vs empirical models as 4x4 heatmaps for FrozenLake."""
    n_states = env.observation_space.n
    grid_size = int(np.sqrt(n_states))  # 4 for 4x4 FrozenLake
    
    # Select one action per state (random or greedy)
    selected_action_per_state = []
    # Randomize 0 or 1
    action_selection = np.random.randint(0, 2)
    for s in range(n_states):
        if action_selection==0:
            # Choose random action
            selected_action_per_state.append(env.action_space.sample())
        else:  
            # Choose current policy
            selected_action_per_state.append(policy.choose_action(s))
    
    # Action arrows for visualization
    action_arrows = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    
    # Get empirical and sampled transition probabilities
    empirical_grids = []
    sampled_grids = []
    reward_grids = []
    
    # Sample a single MDP from the posterior for visualization
    sampled_P, sampled_R = policy.sample_mdp()
    
    for s in range(n_states):
        a = selected_action_per_state[s]
        
        # Empirical probabilities
        counts = np.array([policy.visit_counts.get((s, a, ns), 0) for ns in range(n_states)])
        if counts.sum() > 0:
            empirical_probs = counts / counts.sum()
        else:
            empirical_probs = np.ones(n_states) / n_states
        empirical_grid = empirical_probs.reshape(grid_size, grid_size)
        empirical_grids.append(empirical_grid)
        
        # Sampled probabilities from the pre-sampled MDP
        sampled_probs = sampled_P[s, a, :]
        sampled_grid = sampled_probs.reshape(grid_size, grid_size)
        sampled_grids.append(sampled_grid)
        
        # Get rewards for all (s,a,s_next) transitions from the pre-sampled MDP
        rewards_flat = []
        for s_next in range(n_states):
            reward = sampled_R[s, a, s_next]
            rewards_flat.append(reward)
        reward_grid = np.array(rewards_flat).reshape(grid_size, grid_size)
        reward_grids.append(reward_grid)
    
    # Plot side-by-side comparison: 4x8 layout (4 rows, 8 columns)
    fig, axes = plt.subplots(grid_size, 2 * grid_size, figsize=(20, 10))
    
    for s in range(n_states):
        row = s // grid_size
        col = s % grid_size
        a = selected_action_per_state[s]
        
        # Empirical model (left side)
        ax_emp = axes[row, col]
        im_emp = ax_emp.imshow(empirical_grids[s], cmap='Blues', vmin=0, vmax=1)
        ax_emp.set_title(f'S{s} {action_arrows[a]} Empirical')
        ax_emp.set_xticks([])
        ax_emp.set_yticks([])
        
        # Sampled model (right side)
        ax_samp = axes[row, col + grid_size]
        im_samp = ax_samp.imshow(sampled_grids[s], cmap='Reds', vmin=0, vmax=1)
        ax_samp.set_title(f'S{s} {action_arrows[a]} Sampled')
        ax_samp.set_xticks([])
        ax_samp.set_yticks([])
        
        # Add individual transition rewards as text overlay on each grid cell
        for i in range(grid_size):
            for j in range(grid_size):
                reward_val = reward_grids[s][i, j]
                ax_samp.text(j, i, f"{reward_val:.1f}", 
                            ha='center', va='center', color='black', fontweight='bold', fontsize=8)

    plt.suptitle(f'Model Comparison: Empirical (Left) vs Thompson Sampled {"random" if action_selection == 0 else "greedy"} (Right)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_cartpole_greedy_policy(policy, env, save_path=None):
    """
    Plot greedy policy over angle and angular velocity for CartPole.
    
    Args:
        policy: Policy object with Q-function
        env: CartPole environment with finite abstraction wrapper
        save_path (str, optional): Path to save plot
    """
    # Extract properties from policy and environment
    Q = policy.Q
    state_bins = env.state_bins
    state_ranges = env.state_ranges
    n_actions = env.action_space.n
    
    # Create grid over angle and angular velocity
    angles = np.linspace(state_ranges[2][0]-0.1, state_ranges[2][1]+0.1, 21)
    angular_vels = np.linspace(state_ranges[3][0], state_ranges[3][1], 21)
    greedy_policy = np.zeros((len(angles), len(angular_vels)), dtype=int)
    
    # Helper function to discretize state
    def discretize_state(state, state_bins):
        discrete_indices = []
        for s, bins in zip(state, state_bins):
            discrete_indices.append(np.digitize(s, bins))
        dims = tuple(len(bins) + 1 for bins in state_bins)
        discrete_state = np.ravel_multi_index(discrete_indices, dims=dims)
        return discrete_state
    
    for i, theta in enumerate(angles):
        for j, omega in enumerate(angular_vels):
            # Use center position and zero velocity for visualization
            state = [0.0, 0.0, theta, omega]
            ds = discretize_state(state, state_bins)
            greedy_policy[i, j] = np.argmax(Q[ds])
    
    plt.figure(figsize=(8, 6))
    plt.contourf(angular_vels, angles, greedy_policy, cmap='viridis', levels=np.arange(n_actions+1)-0.5)
    plt.xlabel("Angular Velocity")
    plt.ylabel("Angle")
    plt.title("Final Greedy Policy Map")
    cbar = plt.colorbar(ticks=np.arange(n_actions))
    cbar.ax.set_yticklabels([f'Action {a}' for a in range(n_actions)])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_cartpole_Q_evolution(Q_storage, env, eval_interval, save_path=None, 
                              Q_lower_storage=None, Q_upper_storage=None, 
                              policy_update_interval=None):
    """
    Plot evolution of Q-values over training episodes for CartPole.
    
    Args:
        Q_storage: List of Q-function snapshots over training
        env: CartPole environment with finite abstraction wrapper
        eval_interval: Episode interval between Q snapshots
        save_path (str, optional): Path to save plot
        Q_lower_storage (list, optional): Stored Q-lower bounds
        Q_upper_storage (list, optional): Stored Q-upper bounds
        policy_update_interval (int, optional): Episode interval between Q-bounds updates
    """
    # Extract properties from environment
    state_bins = env.state_bins
    state_ranges = env.state_ranges
    
    # Helper function to discretize state
    def discretize_state(state, state_bins):
        discrete_indices = []
        for s, bins in zip(state, state_bins):
            discrete_indices.append(np.digitize(s, bins))
        dims = tuple(len(bins) + 1 for bins in state_bins)
        discrete_state = np.ravel_multi_index(discrete_indices, dims=dims)
        return discrete_state
    
    num_sub = 6  # number of points to sample in angle and angular velocity dimensions
    angles_sub = np.linspace(state_ranges[2][0], state_ranges[2][1], num_sub)
    angular_vels_sub = np.linspace(-1.0, 1.0, num_sub)
    snapshot_count = len(Q_storage)
    episodes_axis = np.arange(eval_interval, eval_interval*(snapshot_count+1), eval_interval)
    n_actions = Q_storage[0].shape[1]
    
    # Create episode axes for Q-bounds if available
    x_bounds = None
    if Q_lower_storage and Q_upper_storage and policy_update_interval:
        num_bounds_updates = len(Q_lower_storage)
        x_bounds = np.arange(num_bounds_updates) * policy_update_interval
    
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    
    fig, axes = plt.subplots(num_sub, num_sub, figsize=(3*num_sub, 3*num_sub), sharex=True, sharey=True)
    for i, theta in enumerate(angles_sub):
        for j, omega in enumerate(angular_vels_sub):
            ax = axes[i, j]
            state = [0.0, 0.0, theta, omega]  # Use center position and zero velocity
            ds = discretize_state(state, state_bins)
            
            for a in range(n_actions):
                color = colors[a % len(colors)]
                
                # Plot Q-values evolution (solid lines)
                q_evolution = [Q_storage[snap][ds, a] for snap in range(snapshot_count)]
                ax.plot(episodes_axis[:len(q_evolution)], q_evolution, 
                       color=color, linestyle='-', linewidth=2, label=f'Action {a}')
                
                # Plot Q-bounds evolution if available (dashed lines with markers)
                if Q_lower_storage and Q_upper_storage and x_bounds is not None:
                    lower_vals = [Q_lower_storage[k][ds, a] for k in range(len(Q_lower_storage))]
                    upper_vals = [Q_upper_storage[k][ds, a] for k in range(len(Q_upper_storage))]
                    ax.plot(x_bounds, lower_vals, color=color, linestyle='--', marker='^', 
                           markersize=3, linewidth=1, alpha=0.7)
                    ax.plot(x_bounds, upper_vals, color=color, linestyle='--', marker='v', 
                           markersize=3, linewidth=1, alpha=0.7)
            
            ax.set_title(f'θ={theta:.3f}, ω={omega:.1f}')
            ax.grid(True)
            if i == 0 and j == 0:
                ax.legend()
    
    plt.suptitle("Evolution of Q-values over Training Episodes (Subsampled Grid)")
    plt.tight_layout(rect=[0,0,1,0.95])
    if save_path:
        plt.savefig(save_path)
    plt.show()


def compute_monte_carlo_statistics(evaluation_returns):
    """
    Compute statistical summary of Monte Carlo evaluation returns.
    
    Args:
        evaluation_returns (list): List of evaluation return arrays from each run
        
    Returns:
        dict: Dictionary with episodes, statistics arrays, and summary info
    """
    # Find common episodes across all runs
    common_episodes = set(evaluation_returns[0][:, 0])
    total_episodes_per_run = [len(run_data) for run_data in evaluation_returns]
    
    for run_data in evaluation_returns[1:]:
        common_episodes = common_episodes.intersection(set(run_data[:, 0]))
    
    # Sort common episodes
    common_episodes = sorted(common_episodes)
    
    # Check if all evaluations are common and warn if not
    if len(common_episodes) < max(total_episodes_per_run):
        print(f"Warning: Only {len(common_episodes)} evaluation points are common across all runs. "
              f"Run lengths vary from {min(total_episodes_per_run)} to {max(total_episodes_per_run)} evaluations.")
    
    # Extract returns for common episodes only
    returns = []
    for run_data in evaluation_returns:
        run_returns = []
        for episode in common_episodes:
            # Find the return value for this episode
            idx = np.where(run_data[:, 0] == episode)[0]
            if len(idx) > 0:
                run_returns.append(run_data[idx[0], 1])
        returns.append(run_returns)
    
    episodes = np.array(common_episodes)
    returns = np.array(returns)
    
    # Compute statistics
    stats = {
        'episodes': episodes,
        'returns': returns,
        'median': np.median(returns, axis=0),
        'mean': np.mean(returns, axis=0),
        'std': np.std(returns, axis=0),
        'percentiles': np.percentile(returns, [25, 75], axis=0),
        'num_runs': len(evaluation_returns),
        'num_episodes': len(episodes)
    }
    
    return stats


def print_single_run_summary(evaluation_records, run_number, milestones=[i / 10.0 for i in range(11)]):
    """
    Print compact summary of a single Monte Carlo run.
    
    Args:
        evaluation_records (np.array): Evaluation records from single run
        run_number (int): Run number for display
        milestones (list): List of milestone percentages to report
    """
    episodes = evaluation_records[:, 0]
    returns = evaluation_records[:, 1]
    
    print(f"Run {run_number} completed. Evaluation milestones: ", end="")
    
    # Show key milestones
    milestone_indices = [int(len(episodes) * p) - 1 for p in milestones]
    milestone_strs = []
    for p, idx in zip(milestones, milestone_indices):
        milestone_strs.append(f"{int(p*100)}%: {returns[idx]:.1f}")
    
    print(" | ".join(milestone_strs))


def plot_monte_carlo_results(evaluation_data, policy_choice, save_path=None, show_plot=True):
    """
    Create results plot for single run or multiple runs with median/percentiles.
    
    Args:
        evaluation_data: Either:
            - Single run: 2D array [episode, avg_return] 
            - Multiple runs: List of 2D arrays from multiple runs
        policy_choice (str): Policy name for plot title
        save_path (str, optional): Path to save plot
        show_plot (bool): Whether to display plot interactively
    """
    plt.figure(figsize=(12, 6))
    
    # Handle single run vs multiple runs
    if isinstance(evaluation_data, list):
        # Multiple runs - compute statistics and plot median + percentiles
        stats = compute_monte_carlo_statistics(evaluation_data)
        plt.plot(stats['episodes'], stats['median'], label="Median", linewidth=2)
        plt.fill_between(stats['episodes'], stats['percentiles'][0], stats['percentiles'][1], 
                         alpha=0.3, label="5-95 Percentile")
        plt.title(f"Evaluation Returns - {policy_choice} ({stats['num_runs']} runs)")
    else:
        # Single run - plot individual line
        episodes, returns = evaluation_data[:, 0], evaluation_data[:, 1]
        plt.plot(episodes, returns, marker='o', linewidth=1.5, markersize=3)
        plt.title(f"Evaluation Returns - {policy_choice} (Single Run)")
    
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_results_comparison(results_folders, experiment_names, colors=None, save_path=None, axis_mode='linear'):
    """
    Plot comparison of multiple experiments, handling both single runs and Monte Carlo runs.
    
    Args:
        results_folders (list): List of paths to result folders
        experiment_names (list): List of names for each experiment (same length as results_folders)
        colors (list, optional): List of colors for each experiment
        save_path (str, optional): Path to save plot (without extension)
        axis_mode (str, optional): Axis scale mode ('linear', 'log', 'symlog')
    """
    from utils import load_evaluation_records, load_monte_carlo_evaluations
    
    if colors is None:
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink']
    
    # Set PDF font settings
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    matplotlib.rcParams['font.serif'] = ['Computer Modern']

    plt.figure(figsize=(8, 5))
    
    for idx, (folder, name) in enumerate(zip(results_folders, experiment_names)):
        base_name = os.path.basename(folder)
        color = colors[idx % len(colors)]
        
        
        # Check if this is a Monte Carlo run
        if base_name.startswith("monte_carlo"):

            records_list = load_monte_carlo_evaluations(folder)

            if records_list is not None and len(records_list) > 0:
                                
                # Compute statistics (percentiles, median) across runs
                stats = compute_monte_carlo_statistics(records_list)
                plt.plot(stats['episodes'], stats['median'], linestyle='-', color=color, label=name)
                plt.fill_between(stats['episodes'], stats['percentiles'][0], stats['percentiles'][1], 
                                color=color, alpha=0.25)
                print(f"{base_name}: median and percentile curves computed (returns shape: {stats['returns'].shape})")
        else:
            eval_data = load_evaluation_records(folder)
            if eval_data is not None and len(eval_data) > 0:
                episodes = eval_data[:, 0]
                avg_returns = eval_data[:, 1]
                plt.plot(episodes, avg_returns, marker='o', linestyle='-', color=color, label=name)
                print(f"{base_name}: Average return plotted.")
    
    plt.xlabel('Training Episodes')
    plt.ylabel('Average Greedy Return')
    plt.legend()
    plt.tight_layout()
    if axis_mode == 'log':
        plt.yscale('log')
    elif axis_mode == 'symlog':
        plt.yscale('symlog', linthresh=10)
        plt.ylim(top=12)

    if save_path:
        # Ensure target directory exists (save_path is used as a path without extension)
        dirpath = os.path.dirname(save_path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        # Save raster and vector formats. Use bbox_inches='tight' to reduce whitespace.
        plt.savefig(f'{save_path}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{save_path}.pdf', format='pdf', bbox_inches='tight')
        # Vector format for lossless scaling (SVG)
        plt.savefig(f'{save_path}.svg', format='svg', bbox_inches='tight')
    
    plt.show()


def _plot_Q_bounds_snapshot_taxi(Q_lower, Q_upper, Q, env, save_path=None):
    """
    Plot Q-bounds snapshot for Taxi environment.
    
    Args:
        Q_lower (np.ndarray): Lower Q-bounds
        Q_upper (np.ndarray): Upper Q-bounds
        Q (np.ndarray): Current Q-values
        env: Taxi environment
        save_path (str, optional): Path to save plot
    """
    #TODO: Implement Taxi-specific Q-bounds visualization
    # Challenge: 4D state space (500 states) requires creative visualization
    # Possible approaches:
    # 1. Aggregate over passenger/destination dimensions
    # 2. Show specific slices (e.g. passenger in taxi, specific destination)
    # 3. Heatmap showing Q-values for taxi positions with passenger states
    print("Taxi Q-bounds snapshot plotting not yet implemented")
    pass


def plot_taxi_greedy_policy(policy, env, save_path=None):
    """
    Plot greedy policy for Taxi environment.
    
    Args:
        policy: Policy object with Q-table
        env: Taxi environment
        save_path (str, optional): Path to save plot
    """
    #TODO: Implement Taxi-specific greedy policy visualization
    # Show optimal actions for taxi positions given different passenger/destination states
    # Could use multiple subplots for different passenger locations
    print("Taxi greedy policy plotting not yet implemented")
    pass


def plot_taxi_Q_bounds_evolution(Q_lower_storage, Q_upper_storage, Q_storage, 
                                 num_episodes_before_policy_update, 
                                 num_episodes_before_evaluation, env_wrapper, 
                                 save_path=None, final_episode=None):
    """
    Plot Q-bounds evolution for critical Taxi states.
    
    Args:
        Q_lower_storage (list): Stored Q-lower bounds over training
        Q_upper_storage (list): Stored Q-upper bounds over training  
        Q_storage (list): Stored Q-values over training
        num_episodes_before_policy_update (int): Policy update frequency
        num_episodes_before_evaluation (int): Q-value evaluation frequency
        env_wrapper: TaxiWrapper instance
        save_path (str, optional): Path to save plot
        final_episode (int, optional): Final episode number
    """
    if not Q_lower_storage or not Q_upper_storage:
        print("No Q-bounds data available for plotting")
        return
    
    # Get critical states from wrapper
    critical_states_info = env_wrapper.get_critical_states()
    
    num_updates = len(Q_lower_storage)
    
    # Create episode axes for bounds and Q-values
    if final_episode is not None and num_updates > 0:
        x_bounds = list(np.arange(num_updates - 1) * num_episodes_before_policy_update) + [final_episode]
    else:
        x_bounds = np.arange(num_updates) * num_episodes_before_policy_update
    
    x_q_values = np.arange(len(Q_storage)) * num_episodes_before_evaluation
    
    # Create 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Action names for legend
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    
    for idx, (state, description) in enumerate(critical_states_info):
        ax = axes[idx]
        
        # Plot each action
        for action in range(6):
            color = colors[action]
            
            # Q-bounds evolution (dashed lines with markers)
            lower_vals = [Q_lower_storage[k][state, action] for k in range(num_updates)]
            upper_vals = [Q_upper_storage[k][state, action] for k in range(num_updates)]
            
            ax.plot(x_bounds, lower_vals, color=color, linestyle='--', marker='^', 
                   markersize=4, linewidth=1, alpha=0.7,
                   label=f'{action_names[action]}' if idx == 0 else "")
            ax.plot(x_bounds, upper_vals, color=color, linestyle='--', marker='v', 
                   markersize=4, linewidth=1, alpha=0.7)
            
            # Q-values evolution (solid lines)
            if Q_storage:
                q_vals = [Q_storage[k][state, action] for k in range(len(Q_storage))]
                ax.plot(x_q_values, q_vals, color=color, linestyle='-', linewidth=2)
        
        # Formatting
        ax.set_title(description, fontsize=10, pad=5)
        ax.set_xlabel("Episodes", fontsize=9)
        ax.set_ylabel("Q-value", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')
    
    plt.suptitle("Q-bounds Evolution for Critical Taxi States", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_taxi_Q_bounds_snapshot(Q_lower, Q_upper, Q, env_wrapper, save_path=None):
    """
    Plot current Q-bounds snapshot for critical Taxi states.
    
    Args:
        Q_lower (np.ndarray): Current Q-lower bounds
        Q_upper (np.ndarray): Current Q-upper bounds  
        Q (np.ndarray): Current Q-values
        env_wrapper: TaxiWrapper instance
        save_path (str, optional): Path to save plot
    """
    # Get critical states from wrapper
    critical_states_info = env_wrapper.get_critical_states()
    
    # Create 3x4 subplot grid
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()
    
    # Action names
    action_names = ['South', 'North', 'East', 'West', 'Pickup', 'Dropoff']
    
    for idx, (state, description) in enumerate(critical_states_info):
        ax = axes[idx]
        
        x = np.arange(6)  # 6 actions
        width = 0.25
        
        # Plot bars for Q_lower, Q, Q_upper
        ax.bar(x - width, Q_lower[state, :], width=width, label='Q_lower', 
               color='C0', alpha=0.7)
        if Q is not None:
            ax.bar(x, Q[state, :], width=width, label='Q', 
                   color='C1', alpha=0.8)
        ax.bar(x + width, Q_upper[state, :], width=width, label='Q_upper', 
               color='C2', alpha=0.7)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([name[:3] for name in action_names], rotation=45, fontsize=8)
        ax.set_title(description, fontsize=10, pad=5)
        ax.set_ylabel("Q-value", fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8)
    
    plt.suptitle("Current Q-bounds for Critical Taxi States", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def visual_run(env, policy):
    """
    Perform a visual run of the policy in the environment.
    
    Args:
        env: The environment instance
        policy: The policy instance
    """
    state, _ = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
        action = policy.get_greedy_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        env.render()
        state = next_state
    env.close()
