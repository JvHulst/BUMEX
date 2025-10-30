"""
Finite abstraction wrapper for continuous environments.

This module provides a gym wrapper that discretizes continuous state spaces
into finite discrete states, enabling tabular RL methods on continuous environments.
"""

import numpy as np
import gymnasium
from gymnasium import spaces


class FiniteAbstractionWrapper(gymnasium.Wrapper):
    """
    Gym wrapper that discretizes continuous state spaces into finite discrete states.
    
    Converts continuous environments into discrete ones by binning the state space,
    allowing tabular RL methods to work on continuous environments like CartPole.
    """
    
    def __init__(self, env, discretization_config):
        """
        Initialize the finite abstraction wrapper.
        
        Args:
            env: Gym environment with continuous state space
            discretization_config: Dictionary containing discretization parameters
                - bins: List of number of bins per state dimension
                - state_ranges: List of [min, max] ranges per state dimension
                - terminal_limits: Dictionary of terminal state limits per environment
                - observation_limits: Dictionary of observation space limits
        """
        super().__init__(env)
        
        self.bins = discretization_config["bins"]
        self.state_ranges = discretization_config["state_ranges"]
        self.terminal_limits = discretization_config.get("terminal_limits", {})
        self.observation_limits = discretization_config.get("observation_limits", {})
        
        # Create bin boundaries for each state dimension
        self.state_bins = self._create_state_bins()
        
        # Calculate total number of discrete states
        self.n_discrete_states = np.prod([len(bins) + 1 for bins in self.state_bins])
        
        # Override observation space to be discrete
        self.observation_space = spaces.Discrete(self.n_discrete_states)
    
    def _create_state_bins(self):
        """
        Create bin boundaries for each state dimension.
        
        Returns:
            List of arrays containing bin boundaries for each dimension
        """
        bins_list = []
        for (low, high), n_bins in zip(self.state_ranges, self.bins):
            # Create n_bins-1 thresholds to produce n_bins discrete bins
            bins_list.append(np.linspace(low, high, n_bins - 1))
        return bins_list
    
    def discretize_state(self, continuous_state):
        """
        Convert continuous state to discrete state index.
        
        Args:
            continuous_state: Continuous state vector
            
        Returns:
            int: Discrete state index
        """
        discrete_indices = []
        for s, bins in zip(continuous_state, self.state_bins):
            discrete_indices.append(np.digitize(s, bins))
        
        # Convert multi-dimensional index to single index
        dims = tuple(len(bins) + 1 for bins in self.state_bins)
        discrete_state = np.ravel_multi_index(discrete_indices, dims=dims)
        return discrete_state
    
    def get_continuous_from_discrete(self, discrete_state):
        """
        Convert discrete state index to continuous state (bin centers).
        
        Args:
            discrete_state (int): Discrete state index
            
        Returns:
            np.ndarray: Continuous state vector at bin centers
        """
        # Get multi-dimensional bin indices
        dims = tuple(len(bins) + 1 for bins in self.state_bins)
        bin_indices = np.unravel_index(discrete_state, dims)

        # Convert bin indices to continuous values (use same logic as old find_center)
        continuous_state = []
        for idx, bins in zip(bin_indices, self.state_bins):
            # bins is the thresholds array created with np.linspace(low, high, n_bins-1)
            # Follow the old code's logic for center computation:
            # - if idx == 0: center = bins[0] - (bins[1] - bins[0]) / 2
            # - elif idx == len(bins): center = bins[-1] + (bins[-1] - bins[-2]) / 2
            # - else: center = (bins[idx-1] + bins[idx]) / 2
            if idx == 0:
                # below first threshold: extrapolate half the first spacing
                if len(bins) >= 2:
                    center = bins[0] - (bins[1] - bins[0]) / 2.0
                else:
                    # fallback to lower bound of configured range
                    low = self.state_ranges[0][0]
                    center = low
            elif idx == len(bins):
                # above last threshold: extrapolate half the last spacing
                if len(bins) >= 2:
                    center = bins[-1] + (bins[-1] - bins[-2]) / 2.0
                else:
                    high = self.state_ranges[0][1]
                    center = high
            else:
                center = 0.5 * (bins[idx-1] + bins[idx])
            continuous_state.append(center)

        return np.array(continuous_state)
    
    def is_terminal_discrete(self, discrete_state):
        """
        Check if a discrete state is terminal based on environment-specific logic.
        
        Args:
            discrete_state: Discrete state index
            
        Returns:
            bool: True if state is terminal
        """
        # Get multi-dimensional bin indices
        dims = tuple(len(bins) + 1 for bins in self.state_bins)
        bin_indices = np.unravel_index(discrete_state, dims)
        
        # For CartPole: check cart position (dim 0) and pole angle (dim 2)
        if hasattr(self.env.unwrapped, 'spec') and 'CartPole' in str(self.env.unwrapped.spec.id):
            return self._is_terminal_cartpole(bin_indices)
        
        # Default: no discrete states are terminal
        return False
    
    def _is_terminal_cartpole(self, bin_indices):
        """
        Check if CartPole discrete state is terminal.
        
        Args:
            bin_indices: Tuple of bin indices for each dimension
            
        Returns:
            bool: True if terminal
        """
        cart_pos_limits = self.terminal_limits.get("cart_position", [-2.4, 2.4])
        pole_angle_limits = self.terminal_limits.get("pole_angle", [-0.2094395, 0.2094395])
        
        cart_obs_limits = self.observation_limits.get("cart_position", [-4.8, 4.8])
        pole_obs_limits = self.observation_limits.get("pole_angle", [-0.418, 0.418])
        
        # Get bin edges for cart position (dim 0) and pole angle (dim 2)
        cart_bin_idx = bin_indices[0]
        angle_bin_idx = bin_indices[2]
        
        # Calculate bin edges
        x_low = (self.state_bins[0][cart_bin_idx - 1] if cart_bin_idx > 0 
                else cart_obs_limits[0])
        x_high = (self.state_bins[0][cart_bin_idx] if cart_bin_idx < len(self.state_bins[0]) 
                 else cart_obs_limits[1])
        
        theta_low = (self.state_bins[2][angle_bin_idx - 1] if angle_bin_idx > 0 
                    else pole_obs_limits[0])
        theta_high = (self.state_bins[2][angle_bin_idx] if angle_bin_idx < len(self.state_bins[2]) 
                     else pole_obs_limits[1])
        
        # Check if entire bin is beyond terminal limits (deterministic)
        if (x_high <= cart_pos_limits[0] or x_low >= cart_pos_limits[1] or
            theta_high <= pole_angle_limits[0] or theta_low >= pole_angle_limits[1]):
            return True

        #TODO: sampling is not compatible with binary storage of terminality used in cartpole_wrapper! Only works if terminal limits exactly match a bin edge.
        
        # If bin partially overlaps terminal limits:
        # sample uniformly inside the bin and mark terminal if the sample is outside limits.
        def sample_beyond_limits(low, high, limits):
            # Only sample if the bin touches or extends beyond the observation limits region
            if low <= limits[0] or high >= limits[1]:
                sample = np.random.uniform(low, high)
                return (sample <= limits[0]) or (sample >= limits[1])
            return False

        # If either cart position or pole angle bin touches the terminal thresholds,
        # perform one uniform sample per dimension and return True if any sample is outside limits.
        if (x_low <= cart_pos_limits[0] or x_high >= cart_pos_limits[1] or
            theta_low <= pole_angle_limits[0] or theta_high >= pole_angle_limits[1]):
            for low, high, limits in [(x_low, x_high, cart_pos_limits), (theta_low, theta_high, pole_angle_limits)]:
                # if one sample is beyond limits, no need to sample further (box constraints)
                if sample_beyond_limits(low, high, limits):
                    return True

        # If all samples are within limits, return False
        return False
    
    def reset(self, **kwargs):
        """Reset environment and return discretized initial state."""
        continuous_state, info = self.env.reset(**kwargs)
        discrete_state = self.discretize_state(continuous_state)
        return discrete_state, info
    
    def step(self, action):
        """
        Take step in environment and return discretized state.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (discrete_state, reward, done, truncated, info)
        """
        continuous_state, reward, done, truncated, info = self.env.step(action)
        discrete_state = self.discretize_state(continuous_state)
        
        # Check if discrete state is terminal
        if not done and not truncated:
            done = self.is_terminal_discrete(discrete_state)
        
        return discrete_state, reward, done, truncated, info
