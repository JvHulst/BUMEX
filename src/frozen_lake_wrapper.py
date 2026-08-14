"""
FrozenLake environment wrapper for BUMEX.

This module provides environment-specific functionality for FrozenLake,
including terminal state detection, transition dynamics, and probability bounds generation.
"""

import numpy as np


class FrozenLakeWrapper:
    """
    Environment wrapper for FrozenLake that encapsulates all environment-specific logic.
    """
    
    def __init__(self, env):
        """
        Initialize FrozenLake wrapper.
        
        Args:
            env: FrozenLake gym environment
        """
        self.env = env
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n
        self.grid_size = int(np.sqrt(self.n_states))
        self.grid = env.unwrapped.desc.astype('U1')
    
    def is_terminal(self, state):
        """
        Check if a state is terminal (Goal or Hole).
        
        Args:
            state (int): State index
            
        Returns:
            bool: True if state is terminal
        """
        i, j = divmod(state, self.grid_size)
        return self.grid[i, j] in ['G', 'H']
    
    def is_goal(self, state):
        """
        Check if a state is a goal state.
        
        Args:
            state (int): State index
            
        Returns:
            bool: True if state is goal
        """
        i, j = divmod(state, self.grid_size)
        return self.grid[i, j] == 'G'
    
    def get_deterministic_next_state(self, state, action):
        """
        Get deterministic next state (intended direction without slipping).
        Understands grid adjacency and boundaries.
        
        Args:
            state (int): Current state
            action (int): Action (0=Left, 1=Down, 2=Right, 3=Up)
            
        Returns:
            int: Next state index
        """
        i, j = divmod(state, self.grid_size)
        
        if action == 0:      # Left
            nj = max(j - 1, 0)
            ni = i
        elif action == 1:    # Down
            ni = min(i + 1, self.grid_size - 1)
            nj = j
        elif action == 2:    # Right
            nj = min(j + 1, self.grid_size - 1)
            ni = i
        elif action == 3:    # Up
            ni = max(i - 1, 0)
            nj = j
        
        return ni * self.grid_size + nj
    
    def get_possible_next_states(self, state, action):
        """
        Get all possible next states due to slippery mechanics.
        
        Args:
            state (int): Current state
            action (int): Action
            
        Returns:
            List[int]: List of possible next states
        """
        # Map action to possible slip directions
        if action == 0:   # left
            directions = [0, 3, 1]  # left, up, down
        elif action == 1: # down
            directions = [1, 0, 2]  # down, left, right
        elif action == 2: # right
            directions = [2, 3, 1]  # right, up, down
        elif action == 3: # up
            directions = [3, 0, 2]  # up, left, right
        
        next_states = []
        for direction in directions:
            next_state = self.get_deterministic_next_state(state, direction)
            next_states.append(next_state)
        
        return next_states
    
    def generate_P_bounds(self, min_intended_probability=0.3):
        """
        Generate transition probability bounds for FrozenLake slippery mechanics.
        
        Args:
            min_intended_probability (float): Assumed minimum probability of intended transition
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: P_lower, P_upper bounds arrays
        """
        from collections import defaultdict

        # Initialization assumes no transitions are possible.
        P_lower = defaultdict(lambda: 0.0)
        P_upper = defaultdict(lambda: 0.0)
        
        # Updates P_lower and P_upper based on possible next states.
        for state in range(self.n_states):
            for action in range(self.n_actions):
                # Get all possible next states (based on slippery mechanics)
                possible_next_states = self.get_possible_next_states(state, action)
                
                # Set upper bounds: all possible transitions have probability upper bound 1
                for next_state in possible_next_states:
                    P_upper[(state, action, next_state)] = 1.0
                
                # Set lower bound: intended direction has minimum probability
                intended_next_state = self.get_deterministic_next_state(state, action)
                P_lower[(state, action, intended_next_state)] = min_intended_probability
        
        return P_lower, P_upper
    
    def generate_reward_bounds(self):
        """
        Generate reward bounds for FrozenLake (deterministic rewards).
        
        Returns:
            Tuple[dict, dict]: R_lower, R_upper bounds dictionaries (s,a,s') -> reward
        """
        #TODO: add support for (incomplete knowledge) probabilistic rewards with the use_probabilistic_rewards flag.

        # FrozenLake has deterministic rewards: 1 when reaching goal, 0 otherwise
        # Initialize all (s,a,s') tuples with perfect reward knowledge
        from collections import defaultdict

        # Default to reward of 0.0
        R_lower = defaultdict(lambda: 0.0)
        R_upper = defaultdict(lambda: 0.0)
        
        # loop through all (s,a) pairs and assign rewards
        for state in range(self.n_states):
            for action in range(self.n_actions):
                if self.is_goal(state):
                    R_lower[(state, action)] = 1.0
                    R_upper[(state, action)] = 1.0

                                
        return R_lower, R_upper
    
    def update_reward_bounds(self, state, action, next_state, observed_reward, R_lower, R_upper):
        """
        Update reward bounds based on observed reward (assumes deterministic rewards).
        
        Args:
            state: Current state
            action: Action taken  
            next_state: Next state
            observed_reward: Observed reward
            R_lower: Lower reward bounds dictionary
            R_upper: Upper reward bounds dictionary
        
        """
        # full reward knowledge already assumed in frozen_lake

        return R_lower, R_upper
    
    def initialize_Q_bounds(self):
        """
        Initialize Q-bounds for FrozenLake environment.
        For FrozenLake: Q_lower = 0, Q_upper = 1 (typical reward bounds).
        
        Returns:
            tuple: (Q_lower, Q_upper) as numpy arrays
        """
        Q_lower = np.zeros((self.n_states, self.n_actions))
        Q_upper = np.ones((self.n_states, self.n_actions))
        print("Using FrozenLake environment defaults (0, 1).")
        return Q_lower, Q_upper
    
    def initialize_reward_model(self, use_probabilistic_rewards):
        """
        Initialize FrozenLake-specific reward model.
        
        Args:
            use_probabilistic_rewards (bool): Whether to use Beta distributions
            
        Returns:
            tuple: (reward_params, deterministic_rewards)
        """
        if use_probabilistic_rewards:
            # Simple uniform priors for FrozenLake binary rewards
            reward_params = np.ones((self.n_states, self.n_actions, 2))
            return reward_params, None
        else:
            return None, {}
    
    def update_reward_model(self, state, action, next_state, reward, is_terminal,
                           reward_params, deterministic_rewards, use_probabilistic_rewards):
        """
        Update FrozenLake reward model with observed experience.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Observed reward (0 or 1)
            is_terminal: Whether episode terminated
            reward_params: Current Beta parameters (if probabilistic)
            deterministic_rewards: Current deterministic rewards (if not probabilistic)
            use_probabilistic_rewards: Whether using probabilistic model
            
        Returns:
            tuple: (updated_reward_params, updated_deterministic_rewards)
        """
        if use_probabilistic_rewards:
            # Binary reward: 1 for success, 0 for failure
            if reward > 0:
                reward_params[state, action, 1] += 1  # Success
            else:
                reward_params[state, action, 0] += 1  # Failure
            
            return reward_params, deterministic_rewards
        else:
            # Store deterministic reward
            deterministic_rewards[(state, action, next_state)] = reward
            return reward_params, deterministic_rewards
    
    def initialize_transition_priors(self):
        """
        Initialize uniform Dirichlet priors for FrozenLake (no structured assumptions).
        
        Returns:
            dict: Uniform Dirichlet parameters as defaultdict with (s,a,s') keys
        """
        from collections import defaultdict
        
        # Use uniform priors for FrozenLake (no specific transition structure)
        return defaultdict(lambda: 1.0)
        return Q_lower, Q_upper
