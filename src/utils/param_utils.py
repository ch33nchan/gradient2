"""Utilities for working with model parameters and gradients."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple


def count_parameters(model: nn.Module) -> int:
    """
    Count total trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameter_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Get statistics about model parameters.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping parameter names to stats
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            stats[name] = {
                'mean': param.data.mean().item(),
                'std': param.data.std().item(),
                'min': param.data.min().item(),
                'max': param.data.max().item(),
                'norm': param.data.norm().item()
            }
    return stats


def get_gradient_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Get statistics about model gradients.

    Args:
        model: PyTorch model

    Returns:
        Dictionary mapping parameter names to gradient stats
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            stats[name] = {
                'mean': param.grad.mean().item(),
                'std': param.grad.std().item(),
                'min': param.grad.min().item(),
                'max': param.grad.max().item(),
                'norm': param.grad.norm().item()
            }
    return stats


def clip_gradients(
    model: nn.Module,
    max_norm: float
) -> float:
    """
    Clip gradients by global norm.

    Args:
        model: PyTorch model
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm
    ).item()


def flatten_parameters(model: nn.Module) -> torch.Tensor:
    """
    Flatten all model parameters into a single vector.

    Args:
        model: PyTorch model

    Returns:
        Flattened parameter vector
    """
    return torch.cat([
        p.data.flatten()
        for p in model.parameters()
        if p.requires_grad
    ])


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    """
    Flatten all model gradients into a single vector.

    Args:
        model: PyTorch model

    Returns:
        Flattened gradient vector
    """
    grads = []
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            grads.append(p.grad.flatten())
        elif p.requires_grad:
            grads.append(torch.zeros_like(p.data.flatten()))
    return torch.cat(grads)


def set_parameters_from_flat(
    model: nn.Module,
    flat_params: torch.Tensor
) -> None:
    """
    Set model parameters from a flattened vector.

    Args:
        model: PyTorch model
        flat_params: Flattened parameter vector
    """
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(
                flat_params[offset:offset + numel].view_as(p.data)
            )
            offset += numel
