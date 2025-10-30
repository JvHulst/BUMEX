"""
Configuration utilities for BUMEX.

This module provides functions to load and validate configuration files.
"""

import json
import os


def load_config(config_name):
    """
    Load configuration from JSON file.
    
    Args:
        config_name (str): Name of config file (e.g., 'frozen_lake')
        
    Returns:
        dict: Configuration dictionary
    """
    # Check for Monte Carlo override config first
    monte_carlo_config = os.environ.get('BUMEX_MONTE_CARLO_CONFIG')
    if monte_carlo_config and os.path.exists(monte_carlo_config):
        with open(monte_carlo_config, "r") as f:
            config = json.load(f)
        return config
    
    # Load default config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', f'{config_name}.json')
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
