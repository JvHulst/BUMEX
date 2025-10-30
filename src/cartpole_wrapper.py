"""
CartPole environment wrapper for BUMEX.

This module provides environment-specific functionality for CartPole with finite state abstraction,
including terminal state detection, transition dynamics, and probability bounds generation.
"""

import numpy as np
import os
import json
import time
import hashlib
from scipy.spatial import ConvexHull
from finite_abstraction_wrapper import FiniteAbstractionWrapper


class CartPoleWrapper:
    """
    Environment wrapper for CartPole that encapsulates all environment-specific logic.
    Works with FiniteAbstractionWrapper to provide BUMEX interface for discretized CartPole.
    """
    
    def __init__(self, env, config):
        """
        Initialize CartPole wrapper.
        
        Args:
            env: CartPole environment wrapped with FiniteAbstractionWrapper
            config: Configuration dictionary containing discretization parameters
        """
        self.env = env
        self.config = config
        self.n_states = env.observation_space.n
        self.n_actions = env.action_space.n

        # Store the original configuration
        self.config = config
        
        # Extract discretization parameters for potential future use
        self.discretization_config = config.get("discretization", {})
        
        # Extract physics parameters
        self.physics_config = config.get("physics", {})
        
        # Extract commonly used config values as attributes for consistent hashing
        self.mass_pole_set = self.physics_config.get("mass_pole_set", [0.05, 0.1, 0.2])
        self.dt = self.physics_config.get("dt", 0.02)
        self.state_ranges = self.discretization_config["state_ranges"]
        self.num_simulation_samples = self.physics_config.get("num_simulation_samples", 10_000_000)
        self.gamma = self.config.get("gamma", 0.99)
        self.shaped_rewards = self.config.get("shaped_rewards", False)
        
        # Base cache directory as subfolder of BUMEX root (relative to project structure)
        # Navigate from src/ up to BUMEX/ then into cartpole_cache/
        project_root = os.path.dirname(os.path.dirname(__file__))  # Go up from src/ to BUMEX/
        self.cache_dir = os.path.join(project_root, "cartpole_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cached data structures (computed on-demand)
        self.P_lower = None
        self.P_upper = None
        self._terminal_states = None
        self._next_states_cache = None

        # Generate terminal states
        self._terminal_states = self.get_terminal_states()

    def _compute_shaped_reward(self, continuous_state):
        """
        Compute shaped reward for a continuous state.
        
        Args:
            continuous_state: Continuous state [x, x_dot, theta, theta_dot]
            
        Returns:
            float: Shaped reward component
        """
        if self.config.get("shaped_rewards", False):
            return -np.linalg.norm(continuous_state, ord=1) * 0.01
        else:
            return 0.0

    # overwrite env method to introduce shaped rewards
    def step(self, action):
        """
        Take a step in the environment with the given action.
        
        Args:
            action: Action to take in the environment
            
        Returns:
            Tuple: (next_state, reward, done, info)
        """
        next_state, reward, done, info = self.env.step(action)
        
        # Apply shaped rewards if configured
        continuous_state = self.env.get_continuous_from_discrete(next_state)
        shaped_reward = self._compute_shaped_reward(continuous_state)
        reward += shaped_reward
        
        return next_state, reward, done, info
    
    def get_config_hash(self, config_type, mass_pole=None, return_config=False):
        """Generate hash (and optionally config dict) for caching based on type."""
        if config_type == "p_data":
            config = {
                "mass_pole": mass_pole,
                "dt": self.dt,
                "state_ranges": self.state_ranges,
                "num_simulation_samples": self.num_simulation_samples
            }
        elif config_type == "p_bounds":
            config = {
                "mass_pole_set": self.mass_pole_set,
                "dt": self.dt,
                "state_ranges": self.state_ranges,
                "discretization": self.discretization_config
            }
        elif config_type == "q_bounds":
            config = {
                "mass_pole_set": self.mass_pole_set,
                "dt": self.dt,
                "state_ranges": self.state_ranges,
                "discretization": self.discretization_config,
                "gamma": self.gamma,
                "shaped_rewards": self.shaped_rewards
            }
        elif config_type == "p_empirical":
            config = {
                "mass_pole_set": [mass_pole],
                "dt": self.dt,
                "state_ranges": self.state_ranges,
                "discretization": self.discretization_config
            }
        else:
            raise ValueError(f"Unknown config type: {config_type}")
        config_str = json.dumps(config, sort_keys=True)
        hash_val = hashlib.md5(config_str.encode()).hexdigest()[:8]
        if return_config:
            return hash_val, config
        return hash_val
    
    def _simulate_physics(self, continuous_state, action, mass_pole=0.1, dt=0.02):
        """
        Simulate CartPole physics for a single time step.
        
        Args:
            continuous_state: Array [x, x_dot, theta, theta_dot] representing current state
            action: Integer (0 or 1) representing action (left or right force)
            mass_pole: Mass of the pole (kg) for uncertainty modeling.
            dt: Time step duration (seconds)
            
        Returns:
            np.ndarray: Next state [x_next, x_dot_next, theta_next, theta_dot_next]
        """

        # Physical constants for CartPole system
        g = 9.8  # gravity (m/s^2)
        mass_cart = 1.0  # mass of the cart (kg)
        total_mass = mass_cart + mass_pole
        length = 0.5  # half-length of the pole (m)
        pole_mass_length = mass_pole * length
        force_mag = 10.0  # magnitude of the force applied to the cart (N)
        friction_cart = 0.0005  # friction coefficient for the cart
        friction_pole = 0.000002  # friction coefficient for the pole

        # Extract state variables
        x, x_dot, theta, theta_dot = continuous_state

        # Determine force based on action
        force = force_mag if action == 1 else -force_mag

        # Compute equations of motion
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        theta_dot_squared = theta_dot ** 2

        temp = (force + pole_mass_length * theta_dot_squared * sin_theta) / total_mass
        theta_acc = (g * sin_theta - cos_theta * temp - friction_pole * theta_dot / pole_mass_length) / \
                    (length * (4.0 / 3.0 - mass_pole * cos_theta ** 2 / total_mass))
        x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass - friction_cart * np.sign(x_dot)

        # Update state using Euler's method
        x_next = x + dt * x_dot
        x_dot_next = x_dot + dt * x_acc
        theta_next = theta + dt * theta_dot
        theta_dot_next = theta_dot + dt * theta_acc

        return np.array([x_next, x_dot_next, theta_next, theta_dot_next])
    
    def _generate_P_data_for_state_action(self, mass_pole, batch_size=1_000_000):
        """
        Generate transition data using Monte Carlo simulation with random state sampling.
        
        Args:
            mass_pole (float): Mass of pole for uncertainty modeling
            batch_size (int): Number of samples per batch for memory management
            
        Returns:
            str: Path to folder containing saved transition data chunks
        """
        # Calculate number of processing batches
        num_batches = int(np.ceil(self.num_simulation_samples / batch_size))
        
        # Setup save folder
        config_hash, p_data_config = self.get_config_hash("p_data", mass_pole, return_config=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        p_data_dir = os.path.join(self.cache_dir, "p_data")
        os.makedirs(p_data_dir, exist_ok=True)
        save_folder = os.path.join(p_data_dir, f"pdata_{config_hash}_{timestamp}")
        os.makedirs(save_folder, exist_ok=True)
        
        print(f"Generating {self.num_simulation_samples:,} physics simulations for mass_pole = {mass_pole} kg (default mass_pole = 0.1 kg)")
        
        for batch_idx in range(num_batches):
            # Determine batch size (last batch might be smaller)
            current_batch_size = batch_size if batch_idx < num_batches - 1 else (self.num_simulation_samples - batch_idx * batch_size)
            
            # Generate samples for this batch
            batch_data = []
            for _ in range(current_batch_size):
                # Sample random continuous state from state ranges
                continuous_state = np.array([np.random.uniform(low, high) for (low, high) in self.state_ranges])
                
                # Sample random action
                action = np.random.randint(0, self.n_actions)
                
                # Simulate physics
                next_continuous_state = self._simulate_physics(continuous_state, action, mass_pole, self.dt)
                
                # Store transition data
                batch_data.append((continuous_state, action, next_continuous_state))
            
            # Save this batch immediately
            data_file = os.path.join(save_folder, f"chunk_{batch_idx:04d}.npy")
            np.save(data_file, np.array(batch_data, dtype=object))
            
            # Progress tracking
            if num_batches > 1:
                completed_samples = (batch_idx + 1) * batch_size if batch_idx < num_batches - 1 else self.num_simulation_samples
                print(f"  Progress: {completed_samples:,}/{self.num_simulation_samples:,} simulations completed")
        
        # Save final config
        p_data_config["config_hash"] = config_hash
        p_data_config["timestamp"] = timestamp
        p_data_config["num_chunks"] = num_batches
        with open(os.path.join(save_folder, "config.json"), "w") as f:
            json.dump(p_data_config, f, indent=2)
        
        print(f"Saved simulation data ({self.num_simulation_samples:,} transitions, {num_batches} files) to cache")
        return save_folder
    
    def is_terminal(self, state):
        """
        Check if a discrete state is terminal.
        
        Args:
            state (int): Discrete state index
            
        Returns:
            bool: True if state is terminal
        """
        # Use the FiniteAbstractionWrapper's terminal detection
        return self.env.is_terminal_discrete(state)
    
        
    def get_terminal_states(self):
        """
        Get set of all terminal states (cached for performance).
        
        Returns:
            set: Set of terminal state indices
        """
        if self._terminal_states is None:
            self._terminal_states = set()
            for s in range(self.n_states):
                if self.is_terminal(s):
                    self._terminal_states.add(s)
        return self._terminal_states
    
    def get_possible_next_states(self, state, action):
        """
        Get all possible next states from physics simulation (cached for performance).
        
        Args:
            state (int): Current discrete state
            action (int): Action
            
        Returns:
            List[int]: List of possible discrete next states
        """
        if self._next_states_cache is None:
            self._build_next_states_cache()
        
        # Return cached result with fallback to empty list
        return self._next_states_cache.get((state, action), [])
    
    def _build_next_states_cache(self):
        """
        Pre-compute mapping from (state, action) to possible next states.
        Uses correct transition validation: both P_lower key exists AND P_upper > 0.
        """
        from collections import defaultdict
        
        self._next_states_cache = defaultdict(list)
        
        # Find all transitions that are both modeled AND possible
        for (s, a, s_next) in self.P_lower.keys():
            # Check if transition is actually possible (P_upper > 0)
            if self.P_upper.get((s, a, s_next), 0) > 0:
                self._next_states_cache[(s, a)].append(s_next)
        
        # Remove duplicates and convert to regular dict
        for key in self._next_states_cache:
            self._next_states_cache[key] = list(set(self._next_states_cache[key]))
        
        self._next_states_cache = dict(self._next_states_cache)
    
    def generate_P_bounds(self):
        """
        Generate transition probability bounds for CartPole using physics simulation.
        
        Implements hierarchical caching:
        1. Try to load existing P_bounds for all mass_pole values
        2. If not found, try to load existing empirical probabilities for a single mass_pole
        3. If not found, try to generate empirical probabilities from existing P_data
        4. If not found, generate new P_data and compute empirical probabilities

        Afterwards, compute convex hull bounds from empirical models and cache all results.

        Returns:
            Tuple[dict, dict]: P_lower, P_upper sparse dictionaries {(s,a,s'): probability}
        """
        print(f"CartPole uncertainty set: mass_pole_set = {self.mass_pole_set} kg")
        
        # Step 1: Try to load existing P_bounds
        p_bounds_config_hash, p_bounds_config = self.get_config_hash("p_bounds", return_config=True)
        self.P_lower, self.P_upper = self._load_P_bounds(p_bounds_config_hash, p_bounds_config)
        if self.P_lower is not None:
            print("Successfully loaded existing transition probability bounds from cache.")
            
            return self.P_lower, self.P_upper
        
        print(f"Computing transition probability bounds from simulation data...")
        empirical_models = []
        
        for mass_pole in self.mass_pole_set:
            # Step 2: Try to load empirical probabilities for a single mass_pole
            emp_config_hash, emp_config = self.get_config_hash("p_empirical", mass_pole, return_config=True)
            P_lower_emp, P_upper_emp = self._load_P_bounds(emp_config_hash, emp_config)
            
            if P_lower_emp is not None:
                # Use cached empirical probabilities (P_lower = P_upper for empirical)
                empirical_probs = P_lower_emp
                print(f"Loaded cached empirical probabilities for mass_pole = {mass_pole} kg")
            else:
                # Step 3: Try to load existing P_data
                p_data_config_hash = self.get_config_hash("p_data", mass_pole)
                data_folder = self._load_P_data(p_data_config_hash)
                if data_folder is None:
                    # Step 4: Generate new simulation data
                    data_folder = self._generate_P_data_for_state_action(mass_pole)

                # Compute empirical probabilities from the data
                empirical_probs = self._compute_empirical_probabilities(data_folder)
                
                # Cache empirical probabilities as single-element P_bounds
                self._save_P_bounds(empirical_probs, empirical_probs, emp_config_hash, emp_config)
                print(f"Cached empirical probabilities for mass_pole = {mass_pole} kg")
            
            empirical_models.append(empirical_probs)
        
        # Compute convex hull bounds
        self.P_lower, self.P_upper = self._compute_convex_hull_bounds(empirical_models)
        
        # Save P_bounds for future use (uses default p_bounds config)
        self._save_P_bounds(self.P_lower, self.P_upper, p_bounds_config_hash, p_bounds_config)
        
        return self.P_lower, self.P_upper
    
    def _load_P_bounds(self, config_hash, config):
        """Load existing P_bounds if available, selecting the most recent match."""
        # Search for matching P_bounds, selecting most recent

        p_bounds_dir = os.path.join(self.cache_dir, "p_bounds")
        if not os.path.exists(p_bounds_dir):
            return None, None
        
        matching_folders = []
        for folder in os.listdir(p_bounds_dir):
            folder_path = os.path.join(p_bounds_dir, folder)
            if os.path.isdir(folder_path):
                config_file = os.path.join(folder_path, "config.json")
                bounds_file = os.path.join(folder_path, "bounds.npy")
                
                if os.path.exists(config_file) and os.path.exists(bounds_file):
                    with open(config_file, "r") as f:
                        saved_config = json.load(f)
                    
                    # Check if config matches
                    if saved_config.get("config_hash") == config_hash:
                        matching_folders.append((folder, saved_config.get("timestamp", ""), folder_path))
        
        if matching_folders:
            # Select most recent by timestamp
            latest_folder = max(matching_folders, key=lambda x: x[1])
            P_lower, P_upper = np.load(os.path.join(latest_folder[2], "bounds.npy"), allow_pickle=True)
            return P_lower, P_upper
        
        return None, None
    
    def _save_P_bounds(self, P_lower, P_upper, config_hash, config):
        """Save P_bounds with configuration."""
        # Create timestamped folder
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        p_bounds_dir = os.path.join(self.cache_dir, "p_bounds")
        os.makedirs(p_bounds_dir, exist_ok=True)
        save_folder = os.path.join(p_bounds_dir, f"pbounds_{config_hash}_{timestamp}")
        os.makedirs(save_folder, exist_ok=True)
        
        # Save bounds and config
        np.save(os.path.join(save_folder, "bounds.npy"), (P_lower, P_upper))
        config["config_hash"] = config_hash
        config["timestamp"] = timestamp
        
        with open(os.path.join(save_folder, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved transition probability bounds to cache in {save_folder}")
    
    def _load_P_data(self, config_hash):
        """Load existing P_data if available, combining all matching datasets."""
        # Search for matching P_data folders and combine all compatible datasets
        p_data_dir = os.path.join(self.cache_dir, "p_data")
        if not os.path.exists(p_data_dir):
            return None
        
        matching_folders = []
        for folder in os.listdir(p_data_dir):
            folder_path = os.path.join(p_data_dir, folder)
            if os.path.isdir(folder_path):
                config_file = os.path.join(folder_path, "config.json")
                
                if os.path.exists(config_file):
                    with open(config_file, "r") as f:
                        saved_config = json.load(f)
                    
                    # Check if config matches
                    if saved_config.get("config_hash") == config_hash:
                        # Verify data files exist
                        data_files = [f for f in os.listdir(folder_path) if f.startswith("chunk_") and f.endswith(".npy")]
                        if data_files:
                            matching_folders.append(folder_path)
        
        if matching_folders:
            # Return the most recent folder for simplicity (could combine all in future if needed)
            return max(matching_folders, key=lambda x: os.path.basename(x).split('_')[-1])
        
        return None
    
    def load_Q_bounds(self):
        """
        Load existing Q_bounds if available.
        Uses hierarchical caching:
        1. Try to load exact match
        2. If not found, try to load compatible discretizations and adapt bounds


        Returns:
            Tuple[dict, dict] or None: Q_lower, Q_upper numpy arrays or None if not found
        """
        print("Searching for cached Q-value bounds...")
        
        # Define config for matching
        config_hash, q_bounds_config = self.get_config_hash("q_bounds", return_config=True)
        
        # Search for Q_bounds, selecting most recent exact match
        q_bounds_dir = os.path.join(self.cache_dir, "q_bounds")
        if not os.path.exists(q_bounds_dir):
            return None
        
        # First try exact match
        matching_folders = []
        for folder in os.listdir(q_bounds_dir):
            folder_path = os.path.join(q_bounds_dir, folder)
            if os.path.isdir(folder_path):
                config_file = os.path.join(folder_path, "config.json")
                bounds_file = os.path.join(folder_path, "q_bounds.npy")
                
                if os.path.exists(config_file) and os.path.exists(bounds_file):
                    with open(config_file, "r") as f:
                        saved_config = json.load(f)
                    
                    if saved_config.get("config_hash") == config_hash:
                        matching_folders.append((folder, saved_config.get("timestamp", ""), folder_path))
        
        if matching_folders:
            # Select most recent by timestamp
            latest_folder = max(matching_folders, key=lambda x: x[1])
            Q_lower, Q_upper = np.load(os.path.join(latest_folder[2], "q_bounds.npy"), allow_pickle=True)
            print("Successfully loaded exact matching Q-value bounds from cache.")
            return Q_lower, Q_upper
        
        # If exact match not found, try compatible discretizations
        print("No exact match found. Searching for compatible discretizations to adapt...")
        return self._load_compatible_Q_bounds(q_bounds_config)

    
    def _load_compatible_Q_bounds(self, target_config):
        """Load Q_bounds from compatible discretization and adapt to current discretization."""
        q_bounds_dir = os.path.join(self.cache_dir, "q_bounds")
        
        matching_folders = []
        for folder in os.listdir(q_bounds_dir):
            folder_path = os.path.join(q_bounds_dir, folder)
            if os.path.isdir(folder_path):
                config_file = os.path.join(folder_path, "config.json")
                bounds_file = os.path.join(folder_path, "q_bounds.npy")
                
                if os.path.exists(config_file) and os.path.exists(bounds_file):
                    with open(config_file, "r") as f:
                        saved_config = json.load(f)
                    
                    # Check if configs are compatible
                    if self._is_compatible_discretization(saved_config, target_config):
                        matching_folders.append((folder, saved_config.get("timestamp", ""), folder_path, saved_config))
        
        if matching_folders:
            # Select most recent by timestamp
            source_folder = max(matching_folders, key=lambda x: x[1])
            Q_lower_source, Q_upper_source = np.load(os.path.join(source_folder[2], "q_bounds.npy"), allow_pickle=True)

            # Adapt to current discretization (bidirectional)
            Q_lower, Q_upper = self._adapt_Q_bounds_discretization(
                Q_lower_source, Q_upper_source, source_folder[3]["discretization"], target_config["discretization"]
            )

            print(f"Successfully loaded and adapted Q-value bounds from compatible discretization: {source_folder[0]}")
            return Q_lower, Q_upper
        
        return None
    
    def _is_compatible_discretization(self, saved_config, target_config):
        """Check if saved discretization is compatible with target."""
        # Must have same physics parameters
        if (saved_config.get("mass_pole_set") != target_config.get("mass_pole_set") or
            saved_config.get("dt") != target_config.get("dt") or
            saved_config.get("state_ranges") != target_config.get("state_ranges")):
            return False
        
        # Any discretization with same physics is compatible (bidirectional adaptation)
        return True
    
    def _adapt_Q_bounds_discretization(self, Q_lower_source, Q_upper_source, source_disc, target_disc):
        """
        Bidirectional adaptation of Q_bounds from source to target discretization.
        Uses the center-mapping approach from the original match_discretized_Q function.
        """
        print("Adapting Q-value bounds between different discretizations...")
        
        # Get target discretization parameters by multiplying elements of bins
        target_n_states = np.prod(target_disc["bins"])
        n_actions = Q_lower_source.shape[1]
        
        # Create target state centers based on target discretization
        target_state_ranges = self.config.get("state_ranges", [[-4.8, 4.8], [-4, 4], [-0.418, 0.418], [-4, 4]])
        target_bins_config = target_disc["bins"]
        
        target_centers = []
        for (low, high), b in zip(target_state_ranges, target_bins_config):
            thresholds = np.linspace(low, high, b-1)  # b-1 thresholds for b bins
            centers = []
            for i in range(b):
                if i == 0:
                    centers.append(low)
                elif i == b-1:
                    centers.append(high)
                else:
                    centers.append((thresholds[i-1] + thresholds[i]) / 2)
            target_centers.append(np.array(centers))
        
        # Generate meshgrid over target discrete bin indices
        grids = np.meshgrid(*[np.arange(c.size) for c in target_centers], indexing='ij')
        
        # Create target state centers list
        target_state_centers = []
        for idx in range(target_n_states):
            center = []
            for d in range(len(target_centers)):
                flat_grid = grids[d].flatten()
                bin_idx = flat_grid[idx]
                center.append(target_centers[d][bin_idx])
            target_state_centers.append(center)
        
        # Create source bin boundaries
        source_bins_config = source_disc["bins"]
        
        source_bins = []
        for (low, high), b in zip(target_state_ranges, source_bins_config):
            source_bins.append(np.linspace(low, high, b-1))  # b-1 thresholds for b bins
        
        # Initialize target Q-bounds arrays
        Q_lower_target = np.zeros((target_n_states, n_actions))
        Q_upper_target = np.ones((target_n_states, n_actions))
        
        # Map each target state to source state and copy Q-values
        for i in range(target_n_states):
            center = target_state_centers[i]
            
            # Discretize center using source bins to get source index
            discrete_indices = []
            for s, bins in zip(center, source_bins):
                discrete_indices.append(np.digitize(s, bins))
            source_dims = tuple(len(bins) + 1 for bins in source_bins)
            source_index = np.ravel_multi_index(discrete_indices, dims=source_dims)
            
            # Handle out-of-bounds indices
            if source_index < Q_lower_source.shape[0]:
                Q_lower_target[i, :] = Q_lower_source[source_index, :]
                Q_upper_target[i, :] = Q_upper_source[source_index, :]
            # If out of bounds, keep default values (0 and 1)
        
        return Q_lower_target, Q_upper_target
    
    def save_Q_bounds(self, Q_lower, Q_upper):
        """Save Q_bounds with configuration."""
        config_hash, q_bounds_config = self.get_config_hash("q_bounds", return_config=True)
        
        # Create timestamped folder
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        q_bounds_dir = os.path.join(self.cache_dir, "q_bounds")
        os.makedirs(q_bounds_dir, exist_ok=True)
        save_folder = os.path.join(q_bounds_dir, f"qbounds_{config_hash}_{timestamp}")
        os.makedirs(save_folder, exist_ok=True)
        
        # Save bounds and config
        np.save(os.path.join(save_folder, "q_bounds.npy"), (Q_lower, Q_upper))
        q_bounds_config["config_hash"] = config_hash
        q_bounds_config["timestamp"] = timestamp
        q_bounds_config["shape"] = Q_lower.shape
        
        with open(os.path.join(save_folder, "config.json"), "w") as f:
            json.dump(q_bounds_config, f, indent=2)
        
        print(f"Saved Q-value bounds (shape {Q_lower.shape}) to cache in {save_folder}")
    
    def _compute_empirical_probabilities(self, data_folder, batch_size=1_000_000):
        """
        Convert raw transition data to empirical transition probabilities.
        
        Args:
            data_folder: Path to folder containing transition data chunks
            batch_size: Processing batch size for memory management
            
        Returns:
            dict: Empirical probabilities {(discrete_state, action, next_discrete_state): probability}
        """
        from collections import defaultdict
        
        # Count transitions
        counts = defaultdict(float)  # (ds, action, ds_next) -> count
        counts_sa = defaultdict(float)  # (ds, action) -> total count
        
        # Get list of chunk files
        data_files = [f for f in os.listdir(data_folder) if f.startswith("chunk_") and f.endswith(".npy")]
        data_files.sort()
        
        # Load config to get total sample count
        config_file = os.path.join(data_folder, "config.json")
        with open(config_file, "r") as f:
            config = json.load(f)
        total_samples = config.get("num_simulation_samples", 0)
        
        print(f"Computing empirical transition probabilities from {total_samples:,} transitions...")
        
        for file_idx, data_file in enumerate(data_files):
            # Load chunk data
            file_path = os.path.join(data_folder, data_file)
            chunk_data = np.load(file_path, allow_pickle=True)
            
            # Process chunk in sub-batches if needed
            num_sub_batches = int(np.ceil(len(chunk_data) / batch_size))
            
            for sub_batch_idx in range(num_sub_batches):
                start_idx = sub_batch_idx * batch_size
                end_idx = min((sub_batch_idx + 1) * batch_size, len(chunk_data))
                batch_data = chunk_data[start_idx:end_idx]
                
                for continuous_state, action, next_continuous_state in batch_data:
                    # Discretize states
                    discrete_state = self.env.discretize_state(continuous_state)
                    next_discrete_state = self.env.discretize_state(next_continuous_state)
                    
                    # Count transitions
                    counts[(discrete_state, action, next_discrete_state)] += 1.0
                    counts_sa[(discrete_state, action)] += 1.0
            
            if len(data_files) > 1:
                print(f"  Processed chunk {file_idx + 1}/{len(data_files)}")
        
        # Convert counts to probabilities
        empirical_probs = {}
        for (ds, action, ds_next), count in counts.items():
            total_count = counts_sa[(ds, action)]
            empirical_probs[(ds, action, ds_next)] = count / total_count if total_count > 0 else 0.0
        
        return empirical_probs
    
    def _compute_convex_hull_bounds(self, empirical_models):
        """
        Compute convex hull bounds across multiple empirical models.
        
        Args:
            empirical_models: List of empirical probability dictionaries
            
        Returns:
            Tuple[dict, dict]: P_lower, P_upper bounds dictionaries
        """
        from collections import defaultdict
        
        # Collect all possible transitions across all models
        all_transitions = set()
        for model in empirical_models:
            all_transitions.update(model.keys())
        
        # Default probability for missing transitions
        default_prob = 1.0 / self.n_states
        
        # Compute bounds for each transition
        P_lower = {}
        P_upper = {}
        
        for transition in all_transitions:
            # Get probability from each model (use default if missing)
            probs = [model.get(transition, default_prob) for model in empirical_models]
            
            # Convex hull bounds: min and max across uncertainty set
            P_lower[transition] = min(probs) - 0.1
            P_upper[transition] = max(probs) + 0.1
            # Ensure bounds are within [0, 1]
            P_lower[transition] = max(0.0, P_lower[transition])
            P_upper[transition] = min(1.0, P_upper[transition])
        
        return P_lower, P_upper
    
    def generate_reward_bounds(self):
        """
        Generate reward bounds for CartPole (deterministic survival rewards).
        Includes reward shaping if configured.
        
        Returns:
            Tuple[dict, dict]: R_lower, R_upper bounds dictionaries (s,a,s') -> reward
        """
        from collections import defaultdict
        
        
        # Compute shaped rewards for all possible transitions if enabled
        R_lower = defaultdict(lambda: base_reward)
        R_upper = defaultdict(lambda: base_reward)
        
        # Pre-compute shaped rewards for all possible transitions
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # Apply reward shaping if configured
                continuous_state = self.env.get_continuous_from_discrete(s)
                shaped_reward = self._compute_shaped_reward(continuous_state)

                if s in self._terminal_states:
                    base_reward = 0.0  # No survival reward if terminal
                    R_lower[(s, a)] = base_reward + shaped_reward
                    R_upper[(s, a)] = base_reward + shaped_reward

                else:
                    # Base survival reward
                    base_reward = 1.0
                    R_lower[(s, a)] = base_reward + shaped_reward
                    R_upper[(s, a)] = base_reward + shaped_reward
        
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
        # For deterministic rewards, set both bounds to the observed reward
        # R_lower[(state, action)] = observed_reward
        # R_upper[(state, action)] = observed_reward

        if R_lower[(state, action)] != observed_reward:
            print(f"Warning: Observed reward {observed_reward} differs from lower bound ({R_lower[(state, action)]}, {R_upper[(state, action)]})")
            # R_lower[(state, action)] = observed_reward

        if R_upper[(state, action)] != observed_reward:
            print(f"Warning: Observed reward {observed_reward} differs from upper bound ({R_lower[(state, action)]}, {R_upper[(state, action)]})")
            # R_upper[(state, action)] = observed_reward

        return R_lower, R_upper
    
    def get_default_Q_bounds(self):
        """
        Provide environment-specific default Q-bounds initialization.
        For CartPole: Q_lower = 0, Q_upper = 1
        
        Returns:
            tuple: (Q_lower, Q_upper) as numpy arrays
        """
        # Calculate maximum possible discounted return in CartPole
        # Assuming maximum episode length of max_episode_length steps and discount factor gamma
        max_episode_length = self.env.spec.max_episode_steps
        max_return = (1 - self.gamma ** max_episode_length) / (1 - self.gamma)

        # empirically setting Q_upper to 1 is often faster
        Q_lower = np.zeros((self.n_states, self.n_actions))
        Q_upper = np.full((self.n_states, self.n_actions), 1.0)
        return Q_lower, Q_upper
    
    def initialize_Q_bounds(self):
        """
        Initialize Q-bounds for CartPole environment.
        Try to load from cache, otherwise return environment defaults.
        
        Returns:
            tuple: (Q_lower, Q_upper) as numpy arrays
        """
        # Try to load existing Q-bounds
        loaded_Q_bounds = self.load_Q_bounds()
        if loaded_Q_bounds is not None:
            return loaded_Q_bounds
        
        # Fall back to environment defaults
        print(f"No cached Q-value bounds found. Using CartPole environment defaults (lower=0, upper=1).")
        return self.get_default_Q_bounds()

    def initialize_reward_model(self, use_probabilistic_rewards):
        """
        Initialize CartPole-specific reward model for PSRL with exact reward knowledge.
        
        Args:
            use_probabilistic_rewards (bool): Whether to use Beta distributions
            
        Returns:
            tuple: (reward_params, deterministic_rewards)
        """
        if use_probabilistic_rewards:
            # For CartPole, we know exact reward structure: +1 per timestep + shaping
            # Initialize with high confidence (high Beta parameters)
            reward_params = np.ones((self.n_states, self.n_actions, 2))

            print('Warning: probabilistic rewards for Thompson sampling are not well implemented')
            
            # Set high confidence for known reward structure
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    # High confidence in positive rewards (survival + shaping)
                    reward_params[s, a] = [1.0, 10.0]  # Optimistic
            
            return reward_params, None
        else:
            # Use deterministic rewards (exact knowledge)
            deterministic_rewards = {}
            
            # Pre-compute exact rewards for all state-action pairs
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    if self.is_terminal(s):
                        base_reward = 0.0  # No reward if terminal
                    else:
                        base_reward = 1.0

                    continuous_state = self.env.get_continuous_from_discrete(s)
                    shaped_reward = self._compute_shaped_reward(continuous_state)
                    exact_reward = base_reward + shaped_reward
                    
                    # Store for all possible next states
                    deterministic_rewards[(s, a)] = exact_reward
            
            return None, deterministic_rewards
    
    def update_reward_model(self, state, action, next_state, reward, is_terminal,
                           reward_params, deterministic_rewards, use_probabilistic_rewards):
        """
        Update CartPole reward model with observed experience.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            reward: Observed reward
            is_terminal: Whether episode terminated
            reward_params: Current Beta parameters (if probabilistic)
            deterministic_rewards: Current deterministic rewards (if not probabilistic)
            use_probabilistic_rewards: Whether using probabilistic model
            
        Returns:
            tuple: (updated_reward_params, updated_deterministic_rewards)
        """
        if use_probabilistic_rewards:
            # Since we know exact reward structure, no updates needed
            # But we could update confidence if desired
            return reward_params, deterministic_rewards
        else:
            # Store exact observed reward (should match our prediction)
            deterministic_rewards[(state, action)] = reward
            return reward_params, deterministic_rewards
    
    def initialize_transition_priors(self):
        """
        Initialize physics-informed Dirichlet priors for CartPole transitions.
        
        Uses physics knowledge to create structured priors that favor
        physically plausible transitions.
        
        Returns:
            dict: Dirichlet parameters as defaultdict with (s,a,s') keys
        """
        from collections import defaultdict
        
        # Initialize with low prior for unseen transitions
        dirichlet_params = defaultdict(lambda: 0.1)
        
        # Use physics simulation to create informed priors
        for s in range(self.n_states):
            for a in range(self.n_actions):
                # Get continuous state
                continuous_state = self.env.get_continuous_from_discrete(s)
                                
                # Create list of 30 neighbouring states generated randomly by sampling around next_continuous
                neighbours = []
                for _i in range(30):
                    state_ranges = self.config['discretization'].get("state_ranges") # form: [[-2.4, 2.4],[-2.0, 2.0],[-0.2094395, 0.2094395],[-2.5, 2.5]]
                    # use state ranges to scale noise
                    noise = np.random.normal(0, 0.05, size=continuous_state.shape) * [state_ranges[i][1]-state_ranges[i][0] for i in range(len(state_ranges))]
                    neighbour_continuous = continuous_state + noise
                    neighbour_discrete = self.env.discretize_state(neighbour_continuous)
                    neighbours.append(neighbour_discrete)
                    dirichlet_params[(s, a, neighbour_discrete)] += 1.0

        
        return dirichlet_params
