"""
Epsilon-greedy exploration policy for BUMEX.

This module implements the epsilon-greedy exploration strategy.
"""

import numpy as np
import random
from collections import defaultdict


class EpsilonGreedyPolicy:
    """
    Epsilon-greedy exploration policy.
    """

    def __init__(self, env, epsilon_initial=1.0, epsilon_final=0.001, num_episodes=1000, gamma=0.99, alpha=0.1, alpha_final=0.1):
        """
        Initialize epsilon-greedy policy.
        
        Args:
            env: OpenAI Gym environment
            epsilon_initial (float): Initial exploration rate
            epsilon_final (float): Final exploration rate
            num_episodes (int): Total number of episodes for decay
            gamma (float): Discount factor
            alpha (float): Learning rate
        """
        self.env = env
        self.epsilon = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = (epsilon_final / epsilon_initial) ** (1/num_episodes)
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_final = alpha_final
        self.alpha_decay = (alpha_final / alpha) ** (1/num_episodes)

        # Initialize Q-table with zeros
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        
        # Initialize visit counts
        self.visit_counts = defaultdict(int)
    
    def choose_action(self, state, Q=None):
        """
        Choose action using epsilon-greedy strategy.
        
        Args:
            state: Current state
            Q (np.ndarray): Current Q-table (optional, uses self.Q if None)
            
        Returns:
            int: Selected action
        """
        if Q is None:
            Q = self.Q
            
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(Q[state])  # Exploit
    
    def update_epsilon(self):
        """Update epsilon for next episode."""
        self.epsilon *= self.epsilon_decay

    def update_alpha(self):
        """Update alpha for next episode."""
        self.alpha *= self.alpha_decay

    def update_q(self, state, action, reward, next_state, done, truncated):
        """
        Update Q-table using Q-learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
            truncated: Episode truncation flag
        """
        # Update visit counts
        self.visit_counts[(state, action, next_state)] += 1
        
        # Update Q-table using proper Q-learning rule
        if done:
            # Terminal state: no future reward
            target = reward
        else:
            # Non-terminal state: include discounted future value
            target = reward + self.gamma * np.max(self.Q[next_state])
        self.Q[state, action] = self.Q[state, action] + self.alpha * (target - self.Q[state, action])
    
    def get_status_info(self):
        """
        Get policy status information for logging.
        
        Returns:
            dict: Dictionary with policy-specific parameters
        """
        return {"Epsilon": f"{self.epsilon:.3f}", "Alpha": f"{self.alpha:.3f}"}
    
    def get_greedy_action(self, state):
        """
        Get greedy action for evaluation.
        
        Args:
            state: Current state
            
        Returns:
            int: Greedy action
        """
        return np.argmax(self.Q[state, :])
