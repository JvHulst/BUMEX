"""
UCB1 policy implementation for BUMEX.

This module implements the UCB1 (Upper Confidence Bound) exploration strategy.
"""

import numpy as np
import random
from collections import defaultdict


class UCB1Policy:
    """
    UCB1 policy implementation.
    
    Uses Upper Confidence Bound strategy for action selection with confidence bounds
    based on visit counts.
    """
    
    def __init__(self, env, c=1.0, num_episodes=1000, gamma=0.99, alpha=0.1, alpha_final=0.1):
        """
        Initialize UCB1 policy.
        
        Args:
            env: OpenAI Gym environment
            c: Scaling constant for the confidence bound
            gamma (float): Discount factor
            alpha (float): Learning rate
        """
        self.env = env
        self.c = c
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_final = alpha_final
        self.alpha_decay = (alpha_final / alpha) ** (1 / num_episodes)

        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.visit_counts = defaultdict(int)
        self.action_counts = defaultdict(int)  # (state, action) -> count
        self.state_counts = defaultdict(int)   # state -> total count
    
    def choose_action(self, state):
        """
        Choose action using UCB1 strategy.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Derive action counts and state visits from existing visit_counts
        # If this is the first visit to the state, choose randomly
        if self.state_counts[state] == 0:
            return self.env.action_space.sample()
        
        # Calculate UCB1 values for all actions
        ucb_values = np.zeros(self.env.action_space.n)
        for action in range(self.env.action_space.n):
            if self.action_counts[(state, action)] == 0:
                # Assign high value to unvisited actions
                ucb_values[action] = float('inf')
            else:
                # 'c' hyperparameter is used to scale the confidence bound
                confidence_bound = self.c * np.sqrt(np.log(self.state_counts[state]) / self.action_counts[(state, action)])
                ucb_values[action] = self.Q[state, action] + confidence_bound
        
        return np.argmax(ucb_values)
    
    def update_q(self, state, action, reward, next_state, done, truncated):
        """
        Update Q-table using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            truncated: Episode truncation flag
        """
        # Q-learning update with proper terminal state handling
        old_value = self.Q[state, action]
        if done:
            # Terminal state: no future reward
            target = reward
        else:
            # Non-terminal state: include discounted future value
            target = reward + self.gamma * np.max(self.Q[next_state])
        new_value = old_value + self.alpha * (target - old_value)
        self.Q[state, action] = new_value
        
        # Update visit counts
        self.visit_counts[(state, action, next_state)] += 1
        self.action_counts[(state, action)] += 1
        self.state_counts[state] += 1

    def update_alpha(self):
        """Update alpha using exponential decay."""
        self.alpha = self.alpha * self.alpha_decay
    
    def get_status_info(self):
        """
        Get policy status information for logging.
        
        Returns:
            dict: Dictionary with policy-specific parameters
        """
        return {"UCB1 c": f"{self.c}"}
    
    def get_greedy_action(self, state):
        """
        Get greedy action for evaluation.
        
        Args:
            state: Current state
            
        Returns:
            int: Greedy action
        """
        return np.argmax(self.Q[state, :])
