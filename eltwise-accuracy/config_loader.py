import json
import os
from typing import Dict, List, Any


def load_operations_config(config_type: str) -> Dict[str, Any]:
    """
    Load operations configuration from JSON file.
    
    Args:
        config_type: Either "unary" or "binary"
        
    Returns:
        Dictionary containing operations configuration
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If JSON is invalid
        KeyError: If required keys are missing
    """
    config_file = f"{config_type}_operations.json"
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file {config_file} not found")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Validate required keys
    required_keys = ["operations", "dtype"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required key '{key}' in {config_file}")
    
    return config


def get_operations_list(config_type: str) -> List[str]:
    """
    Get list of operations from configuration file.
    
    Args:
        config_type: Either "unary" or "binary"
        
    Returns:
        List of operation names
    """
    config = load_operations_config(config_type)
    return config["operations"]


def get_dtype(config_type: str) -> str:
    """
    Get data type from configuration file.
    
    Args:
        config_type: Either "unary" or "binary"
        
    Returns:
        Data type string
    """
    config = load_operations_config(config_type)
    return config["dtype"]
