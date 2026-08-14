"""
Novel exploring policy with Q-bounds for BUMEX.

This module implements our novel uncertainty-aware exploration strategy using Q-bounds.
"""

import numpy as np
import random
from collections import defaultdict

from fast_regularized import solve_batch


class ExploringPolicy:
    """
    Novel exploring policy using Q-bounds and uncertainty quantification.
    """
    
    def __init__(self, env, env_wrapper=None, c=1.0, delta=0.05, epsilon_initial=1.0, epsilon_final=0.001, num_episodes=1000, gamma=0.95, alpha=0.1, alpha_final=0.1):
        """
        Initialize exploring policy.
        
        Args:
            env: OpenAI Gym environment
            env_wrapper: Environment wrapper for environment-specific functionality
            c (float): Regularization parameter
            delta (float): Confidence level
            epsilon_initial (float): Initial exploration rate
            epsilon_final (float): Final exploration rate
            num_episodes (int): Total number of episodes for epsilon decay
            gamma (float): Discount factor
            alpha (float): Learning rate
            alpha_final (float): Final learning rate
        """
        self.env = env
        self.env_wrapper = env_wrapper
        self.c = c
        self.delta = delta
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_decay = (epsilon_final / epsilon_initial) ** (1.0 / num_episodes)
        self.gamma = gamma
        self.alpha = alpha
        self.alpha_final = alpha_final
        self.alpha_decay = (alpha_final / alpha) ** (1.0 / num_episodes)

        # Initialize visit counts
        self.visit_counts = defaultdict(int)
        self.action_counts = defaultdict(int)  # (state, action) -> count
        
        # Initialize transition probability bounds
        if not self.env_wrapper:
            raise ValueError("Environment wrapper is required for ExploringPolicy")
        self.P_lower, self.P_upper = self.env_wrapper.generate_P_bounds()
        
        # Initialize reward bounds
        self.R_lower, self.R_upper = self.env_wrapper.generate_reward_bounds()
        
        # Initialize Q-bounds using wrapper's environment-specific logic
        self.Q_lower, self.Q_upper = self.env_wrapper.initialize_Q_bounds()
        
        # Initialize Q-table 
        self.Q = np.random.uniform(low=self.Q_lower, high=self.Q_upper, size=(env.observation_space.n, env.action_space.n))

        # Initial update of Q-bounds
        self.update_Q_bounds()
    
    def choose_action(self, state):
        """
        Choose action using exploring policy strategy.
        
        Args:
            state: Current state
            
        Returns:
            int: Selected action
        """
        actions = list(range(self.env.action_space.n))
        weights = []
        for a in actions:
            w = self._phi(state, a)
            weights.append(w)
        weights = np.array(weights)
        
        if random.uniform(0, 1) < self.epsilon:
            if weights.sum() > 0:
                probs = weights / weights.sum()
                return np.random.choice(actions, p=probs)
            else:
                # Fall back to a random action if all weights are zero.
                return self.env.action_space.sample()
        else:
            # Greedy action
            return self.get_greedy_action(state)
    
    def get_greedy_action(self, state):
        """
        Get greedy action for evaluation.
        
        Args:
            state: Current state
            
        Returns:
            int: Greedy action
        """
        return np.argmax(self.Q[state, :])
    
    def update_q(self, state, action, reward, next_state, terminated, truncated):
        """
        Update Q-table using Q-learning rule.
        """

        if hasattr(self.env_wrapper, 'config') and hasattr(self.env_wrapper, '_compute_shaped_reward') and hasattr(self.env, 'get_continuous_from_discrete'):
            if self.env_wrapper.config.get("shaped_rewards", False):
                continuous_state = self.env.get_continuous_from_discrete(state)
                shaped_reward = self.env_wrapper._compute_shaped_reward(continuous_state)
                reward += shaped_reward

        # Standard Q-learning update
        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[next_state, :])
        self.Q[state, action] = self.Q[state, action] + self.alpha * (target - self.Q[state, action])
        
        # Update visit counts
        self.visit_counts[(state, action, next_state)] += 1
        self.action_counts[(state, action)] += 1
        
        # Update reward bounds based on observed reward
        self.R_lower, self.R_upper = self.env_wrapper.update_reward_bounds(state, action, next_state, reward, self.R_lower, self.R_upper)

    def update_epsilon(self):
        """Update epsilon using exponential decay."""
        self.epsilon = self.epsilon * self.epsilon_decay

    def update_alpha(self):
        """Update alpha using exponential decay."""
        self.alpha = self.alpha * self.alpha_decay

    def get_status_info(self):
        """Return status information for logging."""
        return {"epsilon": f"{self.epsilon:.4f}", "alpha": f"{self.alpha:.4f}"}

    def _beta(self, state, action):
        """
        Computes beta(x,u) = (( max(0, Q_upper(x,u) - V_lower(x) ) )^2) / ( 2*(Q_upper(x,u) - Q_lower(x,u)) )
        where V_lower(x) = max_u Q_lower(x,u).
        """
        actions = range(self.env.action_space.n)
        v_lower = max(self.Q_lower[state, a] for a in actions)
        diff = self.Q_upper[state, action] - self.Q_lower[state, action]
        if diff == 0:
            return 0.0
        num = max(0, self.Q_upper[state, action] - v_lower)**2
        return num / (2 * diff)
    
    def _phi(self, state, action):
        """
        Computes the weight for action u in state x according to:
            φ(x,u) = 1, if Q_lower(x,u) >= max_{v ≠ u} Q_lower(x,v)
            φ(x,u) = beta(x,u), if Q_lower(x,u) != Q_upper(x,u) and Q_upper(x,u) > V_lower(x)
            φ(x,u) = 0, otherwise.
        """
        actions = list(range(self.env.action_space.n))
        # Compute V_lower(x)
        V_lower = max(self.Q_lower[state, a] for a in actions)
        # First condition: guaranteed optimal if its lower bound is at least as high as all others.
        others = [self.Q_upper[state, a] for a in actions if a != action]
        max_others = max(others) if others else -np.inf
        if self.Q_lower[state, action] >= max_others:
            return 1.0
        # Second condition: if not guaranteed suboptimal and still uncertain
        if (self.Q_lower[state, action] != self.Q_upper[state, action]) and (self.Q_upper[state, action] > V_lower):
            return self._beta(state, action)
        # Otherwise (guaranteed suboptimal or completely certain but not guaranteed optimal), assign zero weight. This value is the 'zeta' varialbe in the paper
        return 1e-8  # Small positive value to keep exploring all actions in case true model is not within regularized bounds.
    
    def _compute_lambda(self, default_lambda=0.0):
        """
        Compute a data-driven regularization parameter lambda for each (state, action) pair.
        """
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
                
        # Compute the (s,a)-independent logarithmic factor
        log_factor = np.log((n_states * n_actions) / self.delta)
        
        # Initialize the lambda array
        lambdas = np.empty((n_states, n_actions))
        
        # Compute lambda for each (state, action)
        for s in range(n_states):
            for a in range(n_actions):
                N_sa = self.action_counts[(s, a)]
                if N_sa > 0:
                    lambdas[s, a] = self.c / (np.sqrt(log_factor / N_sa))
                else:
                    # If no data is available, use a default low regularization value.
                    lambdas[s, a] = default_lambda

        return lambdas
    
    def _bound_structure(self):
        """
        Index the possible next states of every non-terminal (state, action).

        Returns the state and action of each pair, the indices of its possible
        next states padded into a rectangular array, the mask marking the real
        entries, and the transition probability bounds laid out the same way.

        The transition structure and the probability bounds follow from P_lower
        and P_upper, which are fixed for the whole run, so this is built once and
        reused by every later bound update.
        """
        if getattr(self, '_cached_structure', None) is not None:
            return self._cached_structure

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        is_terminal = self.env_wrapper.is_terminal

        pairs, next_states = [], []
        for s in range(n_states):
            if is_terminal(s):
                continue
            for a in range(n_actions):
                ns_list = self.env_wrapper.get_possible_next_states(s, a)

                # A pair the model set never reached gets a single successor from
                # one physics step at the nominal pole mass. That successor
                # carries no model uncertainty, unlike the rest of the bounds,
                # which are built from the whole mass_pole_set.
                if len(ns_list) == 0 and "CartPole" in self.env.unwrapped.spec.id:
                    s_cont = self.env.get_continuous_from_discrete(s)
                    ns_list = [self.env.discretize_state(
                        self.env_wrapper._simulate_physics(s_cont, a))]
                if len(ns_list) == 0:
                    continue
                pairs.append((s, a))
                next_states.append(list(ns_list))

        width = max(len(ns) for ns in next_states)
        ns_index = np.zeros((len(pairs), width), dtype=np.intp)
        mask = np.zeros((len(pairs), width), dtype=bool)
        p_lower = np.zeros((len(pairs), width))
        p_upper = np.zeros((len(pairs), width))
        for row, ((s, a), ns_list) in enumerate(zip(pairs, next_states)):
            k = len(ns_list)
            ns_index[row, :k] = ns_list
            mask[row, :k] = True
            p_lower[row, :k] = [self.P_lower.get((s, a, ns), 0.0) for ns in ns_list]
            p_upper[row, :k] = [self.P_upper.get((s, a, ns), 1.0) for ns in ns_list]

        states = np.array([s for s, _ in pairs], dtype=np.intp)
        actions = np.array([a for _, a in pairs], dtype=np.intp)
        self._cached_structure = (states, actions, ns_index, mask, p_lower, p_upper)
        return self._cached_structure

    def update_Q_bounds(self, tol=1e-2, max_iter=500):
        """
        Update Q-bounds using regularized optimization.
        Performs iterative optimization starting from current self.Q_lower and self.Q_upper.

        Each sweep solves the regularized transition optimization for every
        (state, action) pair at once, so a sweep costs two batched solves rather
        than one solver call per pair.
        """
        if not self.env_wrapper:
            raise ValueError("Environment wrapper is required for compute_Q_bounds")

        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n

        states, actions, ns_index, mask, p_lower, p_upper = self._bound_structure()

        # Use current Q-bounds as starting point
        Q_lower = np.copy(self.Q_lower)
        Q_upper = np.copy(self.Q_upper)

        lambdas = self._compute_lambda()
        lam = lambdas[states, actions]

        # Empirical transition distribution per pair, uniform where unvisited.
        counts = np.zeros_like(p_lower)
        for row, (s, a) in enumerate(zip(states, actions)):
            counts[row, mask[row]] = [self.visit_counts.get((s, a, ns), 0)
                                      for ns in ns_index[row, mask[row]]]
        totals = counts.sum(axis=1, keepdims=True)
        uniform = mask / np.maximum(mask.sum(axis=1, keepdims=True), 1)
        p_data = np.where(totals > 0, counts / np.where(totals > 0, totals, 1.0), uniform)

        # Reward bounds are refined during training, so they are read afresh here.
        R_lower = np.array([[self.R_lower[(s, a)] for a in range(n_actions)]
                            for s in range(n_states)])
        R_upper = np.array([[self.R_upper[(s, a)] for a in range(n_actions)]
                            for s in range(n_states)])
        terminal = np.array([self.env_wrapper.is_terminal(s) for s in range(n_states)])
        r_lower = R_lower[states, actions]
        r_upper = R_upper[states, actions]

        # Q-iteration over bounds.
        for it in range(max_iter):
            # First, compute value functions.
            V_lower = np.max(Q_lower, axis=1)
            V_upper = np.max(Q_upper, axis=1)
            V_lower_mat = np.where(mask, V_lower[ns_index], 0.0)
            V_upper_mat = np.where(mask, V_upper[ns_index], 0.0)

            p_opt_lower = solve_batch(p_lower, p_upper, p_data, V_lower_mat, lam, mask, 'min')
            p_opt_upper = solve_batch(p_lower, p_upper, p_data, V_upper_mat, lam, mask, 'max')

            Q_lower_new = np.copy(Q_lower)
            Q_upper_new = np.copy(Q_upper)
            Q_lower_new[states, actions] = r_lower + self.gamma * np.sum(p_opt_lower * V_lower_mat, axis=1)
            Q_upper_new[states, actions] = r_upper + self.gamma * np.sum(p_opt_upper * V_upper_mat, axis=1)

            # Terminal states carry the immediate reward only, with no future
            # value, so they are assigned once and never change afterwards.
            if it == 0 and terminal.any():
                Q_lower_new[terminal] = R_lower[terminal]
                Q_upper_new[terminal] = R_upper[terminal]

            delta = max(np.max(np.abs(Q_lower_new - Q_lower)),
                        np.max(np.abs(Q_upper_new - Q_upper)))
            Q_lower, Q_upper = Q_lower_new, Q_upper_new

            # Progress reporting every 1% of the total iterations
            if it % max(1, max_iter//100) == 0 or it == max_iter - 1:
                print(f"Q-bound iteration {it+1}/{max_iter}, delta={delta:.6g}")

            if delta < tol:
                print('Q-bound iterations stopped early: tolerance reached at iteration', it+1)
                break

            # print statement if it is the last iteration and tolerance is not reached
            if it == max_iter - 1:
                print(f"Warning: Q-bound iterations reached max_iter={max_iter} without convergence (delta={delta:.6f})")

        # Save optimized Q-bounds if significant computation was performed (more than 5 iterations)
        # and we're using a wrapper with a save_Q_bounds method and all lambda values are 0
        if it > 5 and hasattr(self.env_wrapper, 'save_Q_bounds') and not np.any(lambdas > 0):
            print(f"Saving optimized Q-bounds after {it+1} iterations...")
            self.env_wrapper.save_Q_bounds(Q_lower, Q_upper)

        self.Q_lower = Q_lower
        self.Q_upper = Q_upper

        return Q_lower, Q_upper
