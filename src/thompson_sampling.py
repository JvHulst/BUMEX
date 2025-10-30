"""
Thompson Sampling (PSRL) policy implementation for BUMEX.

This module implements the Thompson Sampling strategy using Dirichlet priors 
for transitions and deterministic rewards.
"""

import numpy as np
import random
from collections import defaultdict


class ThompsonSamplingPolicy:
    """
    Thompson Sampling (PSRL) policy implementation.
    
    Uses Dirichlet priors for transition probabilities and deterministic rewards
    learned from experience. Periodically samples MDPs from posterior and computes
    optimal policies via value iteration.
    """
    
    def __init__(self, env, env_wrapper=None, use_probabilistic_rewards=False, gamma=0.99):
        """
        Initialize Thompson Sampling policy.
        
        Args:
            env: OpenAI Gym environment
            env_wrapper: Environment wrapper for environment-specific functionality
            use_probabilistic_rewards: If True, use Beta distribution for rewards; if False, use deterministic rewards
            gamma (float): Discount factor
            alpha (float): Learning rate (for interface compatibility)
        """
        self.env = env
        self.env_wrapper = env_wrapper
        self.use_probabilistic_rewards = use_probabilistic_rewards
        self.gamma = gamma
        # Note: Thompson Sampling doesn't need Q-learning, but we keep Q for interface compatibility
        self.Q = np.zeros([env.observation_space.n, env.action_space.n])
        
        # Initialize visit counts (consistent across all policies)
        self.visit_counts = defaultdict(int)
        
        # Initialize Thompson Sampling components using environment wrapper
        (self.reward_params, self.deterministic_rewards) = env_wrapper.initialize_reward_model(use_probabilistic_rewards)
        self.transition_priors = env_wrapper.initialize_transition_priors()
        self.current_policy = None
        
        # Initialize stored value function for warm starting value iteration
        self.stored_V = np.zeros(env.observation_space.n)
    
    @property
    def dirichlet_params(self):
        """Get Dirichlet parameters from visit counts + environment-specific priors."""
        from collections import defaultdict
        
        # Start with environment-specific structured priors
        params = defaultdict(lambda: 1.0)
        params.update(self.transition_priors)
        
        # Add visit counts
        for (s, a, ns), count in self.visit_counts.items():
            params[(s, a, ns)] += count
            
        return params
    
    def choose_action(self, state):
        """
        Choose action according to current sampled policy, resampling periodically.
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Policy will be resampled in the main training loop using existing logic
        if self.current_policy is None:
            self.current_policy = self._sample_mdp_and_compute_policy()
            print(f"Initial MDP and policy sampled.")
        
        return self.current_policy[state]
    
    def update_q(self, state, action, reward, next_state, done, truncated):
        """
        Update Thompson Sampling parameters (Q-learning not needed for PSRL).
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode terminated
            truncated: Episode truncated
        """
        # Thompson Sampling doesn't need Q-learning updates, just posterior updates
        self._update_thompson_sampling(state, action, next_state, reward, done)
    
    def _update_thompson_sampling(self, state, action, next_state, reward, is_terminal=False):
        """Update visit counts and reward model using environment wrapper."""
        # Update visit counts
        self.visit_counts[(state, action, next_state)] += 1
        
        # Update reward model using environment wrapper
        (self.reward_params, 
         self.deterministic_rewards) = self.env_wrapper.update_reward_model(
            state, action, next_state, reward, is_terminal,
            self.reward_params, self.deterministic_rewards, self.use_probabilistic_rewards
        )
    
    def _is_terminal_state(self, state):
        """
        Check if a state is terminal using environment wrapper.
        """
        if not self.env_wrapper:
            raise ValueError("Environment wrapper is required for terminal state detection")
        return self.env_wrapper.is_terminal(state)

    def sample_mdp(self):
        """Sample an MDP from the posterior distribution."""
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        
        # Get Dirichlet parameters
        dirichlet_dict = self.dirichlet_params
        
        # Group transitions by (s,a) pairs for efficient sampling
        sa_groups = defaultdict(list)
        for (s, a, s_next), alpha in dirichlet_dict.items():
            sa_groups[(s, a)].append((s_next, alpha))
        
        # Sample transition probabilities as defaultdict
        sampled_P = defaultdict(lambda: 0.0)
        for (s, a), transitions in sa_groups.items():
            s_next_list, alpha_vec = zip(*transitions)
            sampled_probs = np.random.dirichlet(alpha_vec)
            for prob, s_next in zip(sampled_probs, s_next_list):
                sampled_P[(s, a, s_next)] = prob
        
        # Sample rewards and compute expected rewards R(s,a)
        sampled_R = np.zeros((n_states, n_actions))
        
        if not self.use_probabilistic_rewards:
            # Check if deterministic_rewards uses (s,a) or (s,a,s') format
            sample_key = next(iter(self.deterministic_rewards.keys()))
            if len(sample_key) == 2:  # (s,a) format
                for (s, a), reward in self.deterministic_rewards.items():
                    sampled_R[s, a] = reward
            else:  # (s,a,s') format - compute expected reward using sampled_P
                for s in range(n_states):
                    for a in range(n_actions):
                        expected_reward = 0.0
                        for s_next in range(n_states):
                            p_transition = sampled_P.get((s, a, s_next), 0.0)
                            reward = self.deterministic_rewards.get((s, a, s_next), 0.0)
                            expected_reward += p_transition * reward
                        sampled_R[s, a] = expected_reward
        else:
            # Sample rewards from Beta distributions
            for s in range(n_states):
                for a in range(n_actions):
                    # Sample reward from Beta distribution for this (s,a) pair (scaled to [0,1])
                    scaled_sampled_reward = np.random.beta(self.reward_params[s, a, 1], self.reward_params[s, a, 0])
                    
                    # Scale back to original reward range (environment-specific)
                    # For Taxi: [0,1] -> [-10, 20], For FrozenLake: [0,1] -> [0,1]
                    if hasattr(self.env.unwrapped, 'spec') and 'Taxi' in str(self.env.unwrapped.spec.id):
                        sampled_reward = scaled_sampled_reward * 30.0 - 10.0  # Scale back to [-10, 20]
                    else:
                        sampled_reward = scaled_sampled_reward  # Keep [0,1] for FrozenLake
                    
                    sampled_R[s, a] = sampled_reward

        return sampled_P, sampled_R, sa_groups

    def _sample_mdp_and_compute_policy(self):
        """Sample an MDP from posterior and compute optimal policy via value iteration."""
        sampled_P, sampled_R, sa_groups = self.sample_mdp()
        # Solve sampled MDP using value iteration (sampled_R is already expected reward R(s,a))
        policy = self._solve_sampled_mdp(sampled_P, sampled_R, sa_groups)
        return policy
    
    def _solve_sampled_mdp(self, P, R, sa_groups, tol=1e-2, max_iter=1000):
        """Solve the sampled MDP using value iteration and return policy."""
        n_states, n_actions = R.shape
        
        # Use stored value function as warm start
        V = self.stored_V.copy()
        
        for iteration in range(max_iter):
            V_new = np.zeros(n_states)
            for s in range(n_states):
                action_values = []
                for a in range(n_actions):
                    # Use pre-computed sa_groups for efficient expected future value computation
                    expected_future_value = 0.0
                    if (s, a) in sa_groups:
                        for s_next, _ in sa_groups[(s, a)]:
                            p_transition = P.get((s, a, s_next), 0.0)
                            expected_future_value += p_transition * V[s_next]
                    else:
                        print(f'Warning: state {s}, action {a} does not have any valid transitions.')
                    
                    expected_value = R[s, a] + self.gamma * expected_future_value
                    action_values.append(expected_value)
                V_new[s] = max(action_values)
            if np.max(np.abs(V_new - V)) < tol:
                print('Thompson Sampling value iteration stopped early: tolerance reached at iteration', iteration)
                break
            V = V_new
        
        # Store converged value function for next warm start
        self.stored_V = V.copy()
        
        # Extract policy
        policy = np.zeros(n_states, dtype=int)
        for s in range(n_states):
            action_values = []
            for a in range(n_actions):
                # Use pre-computed sa_groups for efficient expected future value computation
                expected_future_value = 0.0
                if (s, a) in sa_groups:
                    for s_next, _ in sa_groups[(s, a)]:
                        p_transition = P.get((s, a, s_next), 0.0)
                        expected_future_value += p_transition * V[s_next]
                
                expected_value = R[s, a] + self.gamma * expected_future_value
                action_values.append(expected_value)
            policy[s] = np.argmax(action_values)
        return policy
    
    def update_epsilon(self):
        """
        Thompson Sampling doesn't use epsilon decay, but we keep this method for interface consistency.
        """
        pass
    
    def resample_policy(self):
        """
        Resample MDP from posterior and compute new policy.
        """
        self.current_policy = self._sample_mdp_and_compute_policy()
    
    def get_status_info(self):
        """
        Get policy status information for logging.
        
        Returns:
            dict: Dictionary with policy-specific parameters
        """
        return {"Thompson Sampling": "PSRL"}
    
    def get_greedy_action(self, state):
        """
        Get greedy action for evaluation.
        
        Args:
            state: Current state
            
        Returns:
            int: Greedy action from current sampled policy
        """
        if self.current_policy is not None:
            return self.current_policy[state]
        else:
            # If no policy sampled yet, fall back to Q-table
            return np.argmax(self.Q[state, :])
    