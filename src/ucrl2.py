"""
UCRL2 policy implementation for BUMEX.

This module implements the UCRL2 (Upper Confidence Reinforcement Learning) algorithm.
"""

import os
import numpy as np
import random
from collections import defaultdict


class UCRL2Policy:
    """
    UCRL2 policy implementation.
    
    Uses optimistic models based on confidence bounds and solves optimistic MDP.
    """
    
    def __init__(self, env, c=1.0, delta=0.05, epsilon_initial=0.05, epsilon_final=0.001, num_episodes=1000, gamma=0.99):
        """
        Initialize UCRL2 policy.
        
        Args:
            env: OpenAI Gym environment
            c: Scaling constant for confidence bounds
            delta: Confidence level for the high-probability bound
            epsilon_initial: Initial epsilon for exploration
            epsilon_final: Final epsilon for exploration
            num_episodes: Total number of episodes for epsilon decay
            gamma (float): Discount factor
        """
        self.env = env
        self.c = c
        self.delta = delta
        self.epsilon = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = (epsilon_final / epsilon_initial) ** (1/num_episodes)
        self.gamma = gamma
        # Note: UCRL2 doesn't need Q-learning, but we keep Q for interface compatibility
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        self.visit_counts = defaultdict(int)
        self.action_counts = defaultdict(int)  # (state, action) -> count
        self.reward_means = defaultdict(float)  # (state, action) -> running mean of rewards
        
        # UCRL2 optimistic models
        self.optimistic_P = None
        self.optimistic_R = None
        self.optimistic_Q = None
    
    def choose_action(self, state):
        """
        Choose action using UCRL2 strategy with epsilon exploration.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # epsilon-greedy exploration on top of UCRL2
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            return self.get_greedy_action(state)
    
    def update_q(self, state, action, reward, next_state, done, truncated):
        """
        Update visit counts and reward observations (UCRL2 doesn't need Q-learning updates).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode terminated
            truncated: Episode truncated
        """
        # Update visit counts
        self.visit_counts[(state, action, next_state)] += 1
        self.action_counts[(state, action)] += 1
        
        # Update running mean of rewards
        n = self.action_counts[(state, action)]
        old_mean = self.reward_means[(state, action)]
        self.reward_means[(state, action)] = old_mean + (reward - old_mean) / n
    
    def update_epsilon(self):
        """
        Update epsilon using decay schedule (same as epsilon-greedy).
        """
        self.epsilon = self.epsilon * self.epsilon_decay
    
    def update_optimistic_models(self):
        """
        Update optimistic transition and reward models, then solve optimistic MDP.
        """
        self.optimistic_P = self._build_optimistic_transition_model()
        self.optimistic_R = self._build_optimistic_reward_model()
        self.optimistic_Q = self._solve_optimistic_mdp()
    
    def _build_optimistic_transition_model(self):
        """Build optimistic transition model for UCRL2."""
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        P_opt = np.zeros((n_states, n_actions, n_states))
        
        for s in range(n_states):
            for a in range(n_actions):
                # Calculate empirical transition probabilities
                total_visits = self.action_counts[(s, a)]
                if total_visits == 0:
                    # Uniform distribution if no visits
                    P_opt[s, a, :] = 1.0 / n_states
                else:
                    p_hat = np.zeros(n_states)
                    for ns in range(n_states):
                        p_hat[ns] = self.visit_counts.get((s, a, ns), 0) / total_visits
                    
                    # UCRL2 confidence radius
                    confidence_radius = self.c * np.sqrt((2*np.log(2*n_states*n_actions*total_visits/self.delta))/total_visits)
                    confidence_radius = min(confidence_radius, 1.0)  # Cap at 1
                    
                    # Build optimistic probabilities (simplified approach)
                    P_opt[s, a, :] = p_hat + confidence_radius / n_states
                    P_opt[s, a, :] /= P_opt[s, a, :].sum()  # Normalize
        
        return P_opt
    
    def _build_optimistic_reward_model(self):
        """Build optimistic reward model for UCRL2."""
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        R_opt = np.zeros((n_states, n_actions))
        
        for s in range(n_states):
            for a in range(n_actions):
                total_visits = self.action_counts[(s, a)]
                
                if total_visits == 0:
                    R_opt[s, a] = 0.0
                else:
                    # Build confidence bound based on empirical mean
                    empirical_reward = self.reward_means[(s, a)]
                    confidence_radius = self.c * np.sqrt((2*np.log(2*n_states*n_actions*total_visits/self.delta))/total_visits)
                    
                    # Optimistic reward estimate
                    optimistic_reward = empirical_reward + confidence_radius
                    
                    R_opt[s, a] = optimistic_reward
        
        return R_opt
    
    def _solve_optimistic_mdp(self, tol=1e-1, max_iter=200):
        """Solve the optimistic MDP using value iteration."""
        n_states, n_actions = self.optimistic_R.shape
        Q = np.zeros((n_states, n_actions))

        for iteration in range(max_iter):
            Q_new = np.zeros((n_states, n_actions))
            for s in np.random.permutation(n_states):
                for a in np.random.permutation(n_actions):
                    expected_value = np.sum(self.optimistic_P[s, a, :] * np.max(Q, axis=1))
                    Q_new[s, a] = self.optimistic_R[s, a] + self.gamma * expected_value
            
            if np.max(np.abs(Q_new - Q)) < tol:
                print(f"Early stopping condition reached. Solved new optimistic MDP in {iteration} iterations.")
                break
            Q = Q_new
        
        return Q
    
    def get_status_info(self):
        """
        Get policy status information for logging.
        
        Returns:
            dict: Dictionary with policy-specific parameters
        """
        return {"UCRL2 c": f"{self.c}", "epsilon": f"{self.epsilon:.3f}"}
    
    def get_greedy_action(self, state):
        """
        Get greedy action for evaluation.
        
        Args:
            state: Current state
            
        Returns:
            int: Greedy action
        """
        if self.optimistic_Q is not None:
            return np.argmax(self.optimistic_Q[state, :])
        else:
            return self.env.action_space.sample()
