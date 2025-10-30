"""
Novel exploring policy with Q-bounds for BUMEX.

This module implements our novel uncertainty-aware exploration strategy using Q-bounds.
"""

import numpy as np
import cvxpy as cp
import random
from collections import defaultdict


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
        
        # Clip Q-values to stay within bounds
        # self.Q[state, action] = max(self.Q[state, action], self.Q_lower[state, action])
        # self.Q[state, action] = min(self.Q[state, action], self.Q_upper[state, action])
        
    
    def update_epsilon(self):
        """Update epsilon using exponential decay."""
        self.epsilon = self.epsilon * self.epsilon_decay

    def update_alpha(self):
        """Update alpha using exponential decay."""
        self.alpha = self.alpha * self.alpha_decay

    def get_status_info(self):
        """Return status information for logging."""
        return {"epsilon": f"{self.epsilon:.4f}", "alpha": f"{self.alpha:.4f}"}

    def update_Q_bounds_periodic(self):
        """Update Q-bounds using current visit counts."""
        self.Q_lower, self.Q_upper = self.update_Q_bounds(tol=1e-1)
        
        # Clip current Q-values to updated bounds
        self.Q = np.maximum(self.Q, self.Q_lower)
        self.Q = np.minimum(self.Q, self.Q_upper)
    
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
    
    def _regularized_probability_transition_optimization(self, lower_vec, upper_vec, p_data_vec, V_vec, lambda_reg, opt_type='min', solver=cp.MOSEK):
        """
        Solves for the optimal next-state probability distribution p using regularization.
        For opt_type='min': minimizes sum(p*V) + lambda_reg * KL(p_data || p)
        For opt_type='max': maximizes sum(p*V) - lambda_reg * sum(p_data * log(p))
        Returns the optimal p.

        Args:
            lower_vec (np.ndarray): Lower bounds on transition probabilities
            upper_vec (np.ndarray): Upper bounds on transition probabilities
            p_data_vec (np.ndarray): Empirical transition probabilities
            V_vec (np.ndarray): Value vector for next possible states
            lambda_reg (float): Regularization parameter
        opt_type (str): 'min' for minimization, 'max' for maximization
        solver (str): CVXPY solver to use
        
        """
        n = len(V_vec)

        if not opt_type in ['min', 'max']:
            raise ValueError(f"Unknown optimization type: {opt_type}")

        # Use closed-form solution if lambda_reg is 0
        if lambda_reg == 0:
            p_opt = lower_vec.copy()
            if opt_type == 'min':
                sorted_indices = np.argsort(V_vec)
            else:
                sorted_indices = np.argsort(V_vec)[::-1]
            
            for i in range(n):
                if np.sum(p_opt) >= 1:
                    break
                remaining_budget = 1 - np.sum(p_opt)
                p_opt[sorted_indices[i]] += min(remaining_budget, upper_vec[sorted_indices[i]] - lower_vec[sorted_indices[i]])
            return p_opt

        p = cp.Variable(n)
        xi = 1e-8   # small positive value to ensure numerical stability
        if opt_type == 'min':
            objective = cp.sum(cp.multiply(p, V_vec)) + lambda_reg * cp.sum(cp.rel_entr(p_data_vec, p))
            prob = cp.Problem(cp.Minimize(objective),
                              [p >= lower_vec, p <= upper_vec, cp.sum(p) == 1, p >= xi])
        else:
            objective = cp.sum(cp.multiply(p, V_vec)) + lambda_reg * cp.sum(cp.multiply(p_data_vec, cp.log(p)))
            prob = cp.Problem(cp.Maximize(objective),
                              [p >= lower_vec, p <= upper_vec, cp.sum(p) == 1, p >= xi])
        try:
            if solver==None:
                prob.solve()
            else:
                prob.solve(solver=solver)
        except Exception as e:
            raise RuntimeError(f"Optimization failed: {e}")
        if p.value is None: 
            raise RuntimeError("Optimization solver returned None - no feasible solution found")
        p_opt = p.value

        if False:
            # debugging figure
            p_opt = p.value
            # Create a figure with dual y-axes
            import matplotlib.pyplot as plt

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # First y-axis for probability vectors
            x = np.arange(len(p_opt))
            ax1.plot(x, lower_vec, 'r--', label='lower_vec', linewidth=2)
            ax1.plot(x, upper_vec, 'b--', label='upper_vec', linewidth=2)
            ax1.plot(x, p_data_vec, 'g-', label='p_data_vec', marker='o', markersize=4)
            ax1.plot(x, p_opt, 'k-', label='p_opt', marker='s', markersize=4, linewidth=2)

            ax1.set_xlabel('Index')
            ax1.set_ylabel('Probability', color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            ax1.grid(True, alpha=0.3)

            # Second y-axis for V_vec
            ax2 = ax1.twinx()
            ax2.plot(x, V_vec, 'm-', label='V_vec', marker='^', markersize=4, linewidth=2, alpha=0.7)
            ax2.set_ylabel('Value (V_vec)', color='magenta')
            ax2.tick_params(axis='y', labelcolor='magenta')

            # Combine legends from both axes
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')


            plt.title(f'Transition probabilities {opt_type}imized (lambda_reg={lambda_reg:.4f})')
            plt.tight_layout()
            plt.show()

        return p_opt
    
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
    
    def update_Q_bounds(self, tol=1e-2, max_iter=500):
        """
        Update Q-bounds using regularized optimization.
        Performs iterative optimization starting from current self.Q_lower and self.Q_upper.
        """
        if not self.env_wrapper:
            raise ValueError("Environment wrapper is required for compute_Q_bounds")
            
        n_states = self.env.observation_space.n
        n_actions = self.env.action_space.n
        
        # Use wrapper functions
        is_terminal = self.env_wrapper.is_terminal
        
        # Use current Q-bounds as starting point
        Q_lower = np.copy(self.Q_lower)
        Q_upper = np.copy(self.Q_upper)

        lambdas = self._compute_lambda()

        # Initialize counter for state-action pairs with no transitions
        no_transitions_counter = 0

        # Q-iteration over bounds.
        for it in range(max_iter):
            delta = 0.0
            Q_lower_new = np.copy(Q_lower)
            Q_upper_new = np.copy(Q_upper)
            # First, compute value functions.
            V_lower = np.max(Q_lower, axis=1)
            V_upper = np.max(Q_upper, axis=1)

            #TODO: we can be more clever here. Some state-action pairs are more important to the Q-updates than others. Can we sort them? Can we visit the more important ones more frequently through some kind of weighted sampling?
            for s in np.random.permutation(n_states):
                if is_terminal(s):
                    # Update only the first iteration since subsequent iterations will not change anything for terminal states.
                    if it == 0:
                        # Terminal states: Q-values are just the immediate reward (no future value)
                        for a in range(n_actions):
                            Q_lower_new[s, a] = self.R_lower[(s, a)]
                            Q_upper_new[s, a] = self.R_upper[(s, a)]

                            delta = max(delta, abs(Q_lower_new[s, a] - Q_lower[s, a]), abs(Q_upper_new[s, a] - Q_upper[s, a]))
                            
                    # Already assigned for all actions, so skip to next state
                    continue
                    
                for a in np.random.permutation(n_actions):
                    # Use wrapper to get possible next states
                    ns_list = self.env_wrapper.get_possible_next_states(s, a)

                    # if ns_list is empty, simulate the dynamics to find a next state
                    #TODO: this is a bit of a hack. Perhaps we should compute based on all pole_masses such that we properly account for model uncertainty.
                    if len(ns_list) == 0 and "CartPole" in self.env.unwrapped.spec.id:
                        s_cont = self.env.get_continuous_from_discrete(s)
                        s_next_cont = self.env_wrapper._simulate_physics(s_cont, a)
                        ns = self.env.discretize_state(s_next_cont)
                        ns_list = [ns]
                        no_transitions_counter += 1
                    
                    # Build probability bound vectors.
                    lower_vec = np.array([self.P_lower.get((s, a, ns), 0.0) for ns in ns_list])
                    upper_vec = np.array([self.P_upper.get((s, a, ns), 1.0) for ns in ns_list])
                    counts = np.array([self.visit_counts.get((s, a, ns), 0) for ns in ns_list])
                    if counts.sum() == 0:
                        p_data_vec = np.full(len(ns_list), 1/len(ns_list))
                    else:
                        p_data_vec = counts / counts.sum()
                    
                    # Build value vectors.
                    V_lower_vec = np.array([V_lower[ns] for ns in ns_list])
                    V_upper_vec = np.array([V_upper[ns] for ns in ns_list])
                    
                    # Optimize for lower bound.
                    p_opt_lower = self._regularized_probability_transition_optimization(lower_vec, upper_vec, p_data_vec,
                                                      V_lower_vec, lambdas[s,a], opt_type='min', solver = cp.MOSEK)
                    candidate_lower = self.R_lower[(s, a)] + self.gamma * np.dot(p_opt_lower, V_lower_vec)
                    
                    # Optimize for upper bound.
                    p_opt_upper = self._regularized_probability_transition_optimization(lower_vec, upper_vec, p_data_vec,
                                                      V_upper_vec, lambdas[s,a], opt_type='max', solver = cp.MOSEK)
                    candidate_upper = self.R_upper[(s, a)] + self.gamma * np.dot(p_opt_upper, V_upper_vec)
                    
                    Q_lower_new[s, a] = candidate_lower
                    Q_upper_new[s, a] = candidate_upper
                    
                    delta = max(delta, abs(Q_lower_new[s, a] - Q_lower[s, a]), abs(Q_upper_new[s, a] - Q_upper[s, a]))

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
