"""
Taxi environment wrapper for BUMEX.

This module provides environment-specific functionality for Taxi-v3,
including state encoding/decoding, terminal state detection, and transition dynamics.
"""

import numpy as np
from collections import defaultdict


class TaxiWrapper:
    """
    Environment wrapper for Taxi that encapsulates all environment-specific logic.
    Handles 4D state space: (taxi_row, taxi_col, passenger_location, destination)
    """
    
    def __init__(self, env):
        """
        Initialize Taxi wrapper.
        
        Args:
            env: Taxi gym environment
        """
        self.env = env
        self.n_states = env.observation_space.n  # 500
        self.n_actions = env.action_space.n      # 6
        
        # Taxi environment constants
        self.grid_size = 5  # 5x5 grid
        self.n_locations = 5  # 4 pickup/dropoff locations + 1 "in taxi"
        self.n_destinations = 4  # 4 possible destinations
    
    def decode_state(self, state):
        """
        Convert state integer to (taxi_row, taxi_col, passenger_location, destination).
        Following official Taxi-v3 encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
        
        Args:
            state (int): State index (0-499)
            
        Returns:
            tuple: (taxi_row, taxi_col, passenger_location, destination)
        """
        # Reverse the encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
        destination = state % 4
        state = state // 4
        passenger_location = state % 5
        state = state // 5
        taxi_col = state % 5
        taxi_row = state // 5
        return taxi_row, taxi_col, passenger_location, destination
    
    def encode_state(self, taxi_row, taxi_col, passenger_location, destination):
        """
        Convert (taxi_row, taxi_col, passenger_location, destination) to state integer.
        Following official Taxi-v3 encoding: ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
        
        Args:
            taxi_row (int): Taxi row position (0-4)
            taxi_col (int): Taxi column position (0-4)
            passenger_location (int): Passenger location (0-4, where 4 means "in taxi")
            destination (int): Destination location (0-3)
            
        Returns:
            int: State index
        """
        return ((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 + destination
    
    def is_terminal(self, state):
        """
        Check if a state is terminal (passenger successfully dropped off).
        
        Args:
            state (int): State index
            
        Returns:
            bool: True if state is terminal
        """
        taxi_row, taxi_col, passenger_loc, dest_idx = self.decode_state(state)
        
        # Terminal when passenger is at destination (passenger_loc == dest_idx)
        # This happens after successful dropoff at the correct destination
        return passenger_loc == dest_idx
    
    def is_legal_pickup(self, state):
        """
        Check if pickup action is legal in current state.
        
        Args:
            state (int): Current state
            
        Returns:
            bool: True if pickup is legal
        """
        taxi_row, taxi_col, passenger_location, destination = self.decode_state(state)
        
        # Pickup is legal when:
        # 1. Passenger is not already in taxi (passenger_location != 4)
        # 2. Taxi is at the same location as passenger
        if passenger_location == 4:  # Passenger already in taxi
            return False
            
        # Convert passenger location to grid coordinates
        # Locations: 0=R(0,0), 1=G(0,4), 2=Y(4,0), 3=B(4,3)
        passenger_positions = [(0, 0), (0, 4), (4, 0), (4, 3)]
        if passenger_location < 4:
            pass_row, pass_col = passenger_positions[passenger_location]
            return taxi_row == pass_row and taxi_col == pass_col
        
        return False
    
    def is_legal_dropoff(self, state):
        """
        Check if dropoff action is legal in current state.
        
        Args:
            state (int): Current state
            
        Returns:
            bool: True if dropoff is legal
        """
        taxi_row, taxi_col, passenger_location, destination = self.decode_state(state)
        
        # Dropoff is legal when passenger is in taxi (passenger_location == 4)
        return passenger_location == 4
    
    def is_dropoff_at_destination(self, state):
        """
        Check if dropoff would be at the correct destination.
        
        Args:
            state (int): Current state
            
        Returns:
            bool: True if taxi is at passenger destination
        """
        taxi_row, taxi_col, passenger_location, destination = self.decode_state(state)
        
        # Destination locations: 0=R(0,0), 1=G(0,4), 2=Y(4,0), 3=B(4,3)
        destination_positions = [(0, 0), (0, 4), (4, 0), (4, 3)]
        dest_row, dest_col = destination_positions[destination]
        
        return taxi_row == dest_row and taxi_col == dest_col

    def get_deterministic_next_state(self, state, action):
        """
        Get deterministic next state for given state and action.
        
        Args:
            state (int): Current state
            action (int): Action (0=South, 1=North, 2=East, 3=West, 4=Pickup, 5=Dropoff)
            
        Returns:
            int: Next state index
        """
        taxi_row, taxi_col, passenger_loc, dest_idx = self.decode_state(state)
        
        # Movement actions
        if action <= 3:  # Movement actions (0=South, 1=North, 2=East, 3=West)
            
            # check if movement is blockec by walls
            right_wall_states_list = [(3,0), (4,0), (0,1), (1,1), (3,2), (4,2)]
            left_wall_states_list = [(3,1), (4,1), (0,2), (1,2), (3,3), (4,3)]

            if (taxi_row, taxi_col) in right_wall_states_list and action == 2:  # East
                return state  # Blocked by wall
            if (taxi_row, taxi_col) in left_wall_states_list and action == 3:  # West
                return state  # Blocked by wall
            
            # Calculate intended new position
            if action == 0:  # South
                new_row = min(taxi_row + 1, 4)  # Boundary check
                new_col = taxi_col
            elif action == 1:  # North
                new_row = max(taxi_row - 1, 0)  # Boundary check
                new_col = taxi_col
            elif action == 2:  # East
                new_row = taxi_row
                new_col = min(taxi_col + 1, 4)  # Boundary check
            elif action == 3:  # West
                new_row = taxi_row
                new_col = max(taxi_col - 1, 0)  # Boundary check

            return self.encode_state(new_row, new_col, passenger_loc, dest_idx)
            
        elif action == 4:  # Pickup
            if self.is_legal_pickup(state):
                # Pickup successful: passenger moves to taxi (loc 4)
                return self.encode_state(taxi_row, taxi_col, 4, dest_idx)
            else:
                # Pickup failed: state unchanged
                return state
        elif action == 5:  # Dropoff
            if self.is_legal_dropoff(state):  # Passenger is in taxi
                # Find which location corresponds to current taxi position
                dropoff_locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B
                current_location = None
                for i, (loc_row, loc_col) in enumerate(dropoff_locations):
                    if taxi_row == loc_row and taxi_col == loc_col:
                        current_location = i
                        break
                
                if current_location is not None:
                    # Drop passenger at current taxi location
                    return self.encode_state(taxi_row, taxi_col, current_location, dest_idx)
                else:
                    # Taxi not at a dropoff location, passenger stays in taxi
                    return state
            else:
                # No passenger to dropoff: state unchanged
                return state
        
        return state

    def get_possible_next_states(self, state, action):
        """
        Get all possible next states from current state and action.

        Taxi movement, pickup and dropoff all resolve deterministically, so each
        (state, action) has exactly one successor. The uncertainty the policy
        works with sits in the reward model instead.

        Args:
            state (int): Current state
            action (int): Action

        Returns:
            list: List of possible next states
        """
        return [self.get_deterministic_next_state(state, action)]
    
    def generate_P_bounds(self):
        """
        Generate transition probability bounds for Taxi environment.

        The single successor of each (state, action) gets an upper bound of one.
        Lower bounds stay at the default of zero, so the transition model is
        treated as unknown until the visit counts fill it in.

        Returns:
            tuple: (P_lower, P_upper) probability bound dictionaries
        """
        # Initialize with defaultdict for memory efficiency (default 0.0)
        P_lower = defaultdict(float)
        P_upper = defaultdict(float)

        for state in range(self.n_states):
            for action in range(self.n_actions):
                for next_state in self.get_possible_next_states(state, action):
                    P_upper[(state, action, next_state)] = 1.0

        return P_lower, P_upper
    
    def _get_exact_reward(self, state, action):
        """
        Get exact Taxi-v3 reward for transition.
        
        Args:
            state: Current state
            action: Action taken
            
        Returns:
            int: Exact reward for this transition
        """
        if self.is_terminal(state):
            # if dropped off: no more future rewards
            return 20

        if action < 4:  # Movement actions (0=South, 1=North, 2=East, 3=West)
            return -1  # Always -1 per step
        elif action == 4:  # Pickup
            if self.is_legal_pickup(state):
                return -1  # Legal pickup costs -1 (step penalty)
            else:
                return -10  # Illegal pickup penalty
        elif action == 5:  # Dropoff
            if self.is_legal_dropoff(state):  # Passenger is in taxi
                if self.is_dropoff_at_destination(state):
                    return 20  # Successful dropoff at correct destination (reward obtained in terminal state)
                else:
                    return -1
            else:
                return -10  # No passenger to dropoff

    def generate_reward_bounds(self):
        """
        Generate reward bounds for Taxi environment for the exploring_policy.
        
        Returns:
            tuple: (R_lower, R_upper) reward bound dictionaries (s,a,s') -> reward
        """
        # Use exact rewards for all transitions (default 0)
        R_lower = defaultdict(lambda: -20.0)
        R_upper = defaultdict(lambda: -2.0)
                        
        # Has approximate knowledge about movements.
        # Has exact knowledge about success states and legal pickups/dropoffs.
        # Does not know about illegal pickups/dropoffs
        for state in range(self.n_states):
            for action in range(self.n_actions):
                if self.is_terminal(state):
                    exact_reward = self._get_exact_reward(state, action)
                    R_lower[(state, action)] = exact_reward
                    R_upper[(state, action)] = exact_reward
                elif action < 4:  # Movement actions (0=South, 1=North, 2=East, 3=West)
                    exact_reward = self._get_exact_reward(state, action)
                    R_lower[(state, action)] = exact_reward - 0.5
                    R_upper[(state, action)] = exact_reward + 0.5
                elif action == 4 and self.is_legal_pickup(state):
                    exact_reward = self._get_exact_reward(state, action)
                    R_lower[(state, action)] = exact_reward
                    R_upper[(state, action)] = exact_reward
                elif action == 5 and self.is_legal_dropoff(state):
                    exact_reward = self._get_exact_reward(state, action)
                    R_lower[(state, action)] = exact_reward
                    R_upper[(state, action)] = exact_reward

        return R_lower, R_upper
        
    
    def update_reward_bounds(self, state, action, next_state, observed_reward, R_lower, R_upper):
        """
        Update reward bounds based on observed reward.
        
        Args:
            state: Current state
            action: Action taken
            next_state: Next state
            observed_reward: Observed reward
            R_lower: Lower reward bounds dictionary
            R_upper: Upper reward bounds dictionary
        """
        # For Taxi (deterministic rewards), set both bounds to observed reward
        R_lower[(state, action)] = observed_reward
        R_upper[(state, action)] = observed_reward

        return R_lower, R_upper

    def initialize_Q_bounds(self):
        """
        Initialize Q-bounds for Taxi environment.
        
        Returns:
            tuple: (Q_lower, Q_upper) as numpy arrays
        """
        # Taxi reward structure:
        # - Movement actions: -1 per step
        # - Illegal pickup/dropoff: -10 penalty
        # - Successful dropoff: +20 reward
        # 
        # Worst case: Long episode with many illegal actions
        # Best case: Quick completion with +20 reward

        Q_lower = np.full((self.n_states, self.n_actions), -10.0)
        Q_upper = np.full((self.n_states, self.n_actions), 0.0)
        
        print("Using Taxi environment initial Q-bounds: Q_lower = -10, Q_upper = 0")
        return Q_lower, Q_upper
    
    def get_critical_states(self):
        """
        Get critical states for Q-bounds visualization.
        Returns 12 representative states covering different scenarios.
        
        Returns:
            list: List of (state, description) tuples for visualization
        """
        critical_states = []
        
        # Pickup scenarios (4 states) - taxi at passenger locations
        pickup_locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B locations
        pickup_names = ['R', 'G', 'Y', 'B']
        for i, ((row, col), name) in enumerate(zip(pickup_locations, pickup_names)):
            state = self.encode_state(row, col, i, (i + 1) % 4)  # passenger at location i, dest different
            critical_states.append((state, f"Pickup {name} (taxi at {name}, pass at {name})"))
        
        # Dropoff scenarios (4 states) - taxi at destination with passenger
        dest_locations = [(0, 0), (0, 4), (4, 0), (4, 3)]  # R, G, Y, B locations  
        dest_names = ['R', 'G', 'Y', 'B']
        for i, ((row, col), name) in enumerate(zip(dest_locations, dest_names)):
            state = self.encode_state(row, col, 4, i)  # passenger in taxi, dest i
            critical_states.append((state, f"Dropoff {name} (taxi at {name}, dest {name})"))
        
        # Movement scenarios (4 states) - taxi transporting passenger
        transport_scenarios = [
            ((1, 2), 4, 0, "Center→R"),      # Center position going to R
            ((2, 2), 4, 1, "Center→G"),      # Center position going to G  
            ((3, 1), 4, 2, "Transit→Y"),     # In transit to Y
            ((1, 3), 4, 3, "Transit→B")      # In transit to B
        ]
        for (row, col), pass_loc, dest, desc in transport_scenarios:
            state = self.encode_state(row, col, pass_loc, dest)
            critical_states.append((state, f"Transport {desc} (pass in taxi)"))
        
        return critical_states
    
    def initialize_reward_model(self, use_probabilistic_rewards):
        """
        Initialize Taxi-specific reward model for PSRL with structured priors matching exploring_policy knowledge.
        
        Args:
            use_probabilistic_rewards (bool): Whether to use Beta distributions
            
        Returns:
            tuple: (reward_params, deterministic_rewards)
        """
        if use_probabilistic_rewards:
            # Initialize Beta parameters [failures, successes] for each (s,a) pair
            reward_params = np.ones((self.n_states, self.n_actions, 2))
            
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    if self.is_terminal(s):
                        # Terminal states: exact knowledge (high confidence)
                        reward_params[s, a] = [1.0, 20.0]  # Very optimistic for terminal states
                    elif a < 4:  # Movement actions (0-3)
                        # Approximate knowledge: movement costs around -1 with some uncertainty
                        reward_params[s, a] = [2.0, 1.0]  # Slightly pessimistic
                    elif a == 4 and self.is_legal_pickup(s):
                        # Legal pickup: exact knowledge (like exploring_policy)
                        reward_params[s, a] = [2.0, 1.0]  # Around -1 scaled to [0,1]
                    elif a == 5 and self.is_legal_dropoff(s):
                        # Legal dropoff: exact knowledge (like exploring_policy)
                        if self.is_dropoff_at_destination(s):
                            reward_params[s, a] = [1.0, 20.0]  # Very optimistic (+20)
                        else:
                            reward_params[s, a] = [3.0, 1.0]  # Around -1 scaled to [0,1]
                    else:
                        # Unknown illegal actions: use default uncertain bounds
                        reward_params[s, a] = [4.0, 1.0]  # Slightly pessimistic
            
            return reward_params, None
        else:
            # Use deterministic rewards based on the same structured knowledge as probabilistic version
            deterministic_rewards = {}
            
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    exact_reward = self._get_exact_reward(s, a)
                    if self.is_terminal(s):
                        # Terminal states: use exact reward (20 for successful completion)
                        deterministic_rewards[(s, a)] = exact_reward
                    elif a < 4:  # Movement actions (0-3)
                        # Movement actions: always -1 (exact knowledge)
                        deterministic_rewards[(s, a)] = exact_reward + np.random.uniform(-0.3, 0.3)  # Small noise to add uncertainty
                    elif a == 4 and self.is_legal_pickup(s):
                        # Legal pickup: exact knowledge
                        deterministic_rewards[(s, a)] = exact_reward
                    elif a == 5 and self.is_legal_dropoff(s):
                        # Legal dropoff: exact knowledge
                        deterministic_rewards[(s, a)] = exact_reward
            
            return None, deterministic_rewards
    
    def update_reward_model(self, state, action, next_state, reward, is_terminal, 
                           reward_params, deterministic_rewards, use_probabilistic_rewards):
        """
        Update Taxi reward model with observed experience.
        
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
            # Scale reward from [-10, 20] to [0, 1] range for Beta distribution
            scaled_reward = (reward + 10) / 30.0  # Maps [-10,20] to [0,1]
            
            if scaled_reward > 0.5:  # Success (positive scaled reward)
                reward_params[state, action, 1] += 1
            else:  # Failure (negative scaled reward)
                reward_params[state, action, 0] += 1
            
            # Handle terminal states
            if is_terminal:
                for a in range(self.n_actions):
                    reward_params[next_state, a, 0] += 1  # All actions give 0 reward
            
            return reward_params, deterministic_rewards
        else:
            # Store deterministic reward
            deterministic_rewards[(state, action)] = reward

            return reward_params, deterministic_rewards
    
    def initialize_transition_priors(self):
        """
        Initialize structured Dirichlet priors for Taxi transition probabilities.
        
        Assumes that transitions can only change at most one state component at a time:
        - taxi_row, taxi_col, passenger_location can change
        - destination remains fixed (NOTE: this assumption may not hold if environment 
          has 'rain' or other stochastic effects that could change the destination)
        
        Returns:
            dict: Dirichlet parameters as defaultdict with (s,a,s') keys
        """
        from collections import defaultdict
        
        # Initialize with very low prior (0.01) for impossible transitions
        dirichlet_params = defaultdict(lambda: 0.01)
        
        for state in range(self.n_states):
            taxi_row, taxi_col, pass_loc, dest = self.decode_state(state)
            
            for action in range(self.n_actions):
                for next_state in range(self.n_states):
                    next_taxi_row, next_taxi_col, next_pass_loc, next_dest = self.decode_state(next_state)
                    
                    # Count how many components changed
                    changes = 0
                    if taxi_row != next_taxi_row:
                        changes += 1
                    if taxi_col != next_taxi_col:
                        changes += 1
                    if pass_loc != next_pass_loc:
                        changes += 1
                    if dest != next_dest:
                        changes += 1
                    
                    # Structured prior: reasonable probability for plausible transitions
                    if changes <= 1:  # At most one component changes
                        if dest == next_dest:  # Destination never changes
                            dirichlet_params[(state, action, next_state)] = 1.0  # Plausible transition
                    # Else keep very low prior (0.01) for implausible transitions
        
        return dirichlet_params
