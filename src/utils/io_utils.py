"""Utilities for input/output operations."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict
import torch
import yaml


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.

    Args:
        data: Dictionary to save
        filepath: Path to JSON file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_pickle(data: Any, filepath: str) -> None:
    """
    Save data to pickle file.

    Args:
        data: Object to save
        filepath: Path to pickle file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


def load_pickle(filepath: str) -> Any:
    """
    Load data from pickle file.

    Args:
        filepath: Path to pickle file

    Returns:
        Loaded object
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_yaml(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to YAML file.

    Args:
        data: Dictionary to save
        filepath: Path to YAML file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load data from YAML file.

    Args:
        filepath: Path to YAML file

    Returns:
        Loaded dictionary
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, Any],
    filepath: str
) -> None:
    """
    Save training checkpoint.

    Args:
        model: PyTorch model
        optimizer: PyTorch optimizer
        epoch: Current epoch number
        metrics: Training metrics
        filepath: Path to checkpoint file
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, filepath)


def load_checkpoint(
    filepath: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer = None
) -> Dict[str, Any]:
    """
    Load training checkpoint.

    Args:
        filepath: Path to checkpoint file
        model: PyTorch model to load weights into
        optimizer: Optional PyTorch optimizer to load state into

    Returns:
        Dictionary with epoch and metrics
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return {
        'epoch': checkpoint.get('epoch', 0),
        'metrics': checkpoint.get('metrics', {})
    }
