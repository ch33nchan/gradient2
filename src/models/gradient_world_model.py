"""
Gradient world model: predicts how parameters will change given experiences.

This is the core component enabling gradient hacking - the agent learns
to predict its own gradient updates and can use this to plan manipulations.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class GradientWorldModel(nn.Module):
    """
    Model that predicts gradient updates given current state and experience.

    Takes as input:
    - Current parameter snapshot (or summary)
    - Experience (observation, action, reward, next observation)
    - Training configuration (learning rate, etc.)

    Outputs:
    - Predicted gradient or parameter change
    """

    def __init__(
        self,
        param_dim: int,
        experience_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        """
        Initialize gradient world model.

        Args:
            param_dim: Dimension of parameter representation
            experience_dim: Dimension of experience representation
            hidden_dim: Hidden layer size
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.param_dim = param_dim
        self.experience_dim = experience_dim

        input_dim = param_dim + experience_dim

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, param_dim))

        self.network = nn.Sequential(*layers)

    def forward(
        self,
        param_summary: torch.Tensor,
        experience: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict gradient or parameter change.

        Args:
            param_summary: Current parameter representation (batch, param_dim)
            experience: Experience representation (batch, experience_dim)

        Returns:
            Predicted parameter change (batch, param_dim)
        """
        combined = torch.cat([param_summary, experience], dim=-1)
        return self.network(combined)

    def predict_update(
        self,
        current_params: torch.Tensor,
        experience: torch.Tensor,
        learning_rate: float = 0.01
    ) -> torch.Tensor:
        """
        Predict new parameters after update.

        Args:
            current_params: Current parameter values
            experience: Experience representation
            learning_rate: Learning rate for update

        Returns:
            Predicted new parameters
        """
        grad_prediction = self.forward(current_params, experience)
        new_params = current_params - learning_rate * grad_prediction
        return new_params


class RecurrentGradientModel(nn.Module):
    """
    Recurrent version of gradient world model for sequential prediction.

    Maintains hidden state across multiple gradient predictions.
    """

    def __init__(
        self,
        param_dim: int,
        experience_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1
    ):
        """
        Initialize recurrent gradient model.

        Args:
            param_dim: Dimension of parameter representation
            experience_dim: Dimension of experience representation
            hidden_dim: Hidden state size
            num_layers: Number of GRU layers
        """
        super().__init__()

        self.param_dim = param_dim
        self.experience_dim = experience_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = param_dim + experience_dim

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.output_layer = nn.Linear(hidden_dim, param_dim)

    def forward(
        self,
        param_summary: torch.Tensor,
        experience: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict gradient with recurrent state.

        Args:
            param_summary: Current parameter representation (batch, seq_len, param_dim)
            experience: Experience representation (batch, seq_len, experience_dim)
            hidden: Hidden state (num_layers, batch, hidden_dim)

        Returns:
            Tuple of (predicted gradient, new hidden state)
        """
        if param_summary.dim() == 2:
            param_summary = param_summary.unsqueeze(1)
        if experience.dim() == 2:
            experience = experience.unsqueeze(1)

        combined = torch.cat([param_summary, experience], dim=-1)
        output, hidden = self.gru(combined, hidden)
        grad_prediction = self.output_layer(output)

        return grad_prediction, hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size

        Returns:
            Zero hidden state
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


def create_param_summary(model: nn.Module, summary_dim: int = 64) -> torch.Tensor:
    """
    Create a fixed-size summary of model parameters.

    Args:
        model: PyTorch model
        summary_dim: Target dimension for summary

    Returns:
        Parameter summary vector
    """
    all_params = []
    for param in model.parameters():
        if param.requires_grad:
            all_params.append(param.data.flatten())

    if not all_params:
        return torch.zeros(summary_dim)

    flat_params = torch.cat(all_params)

    if len(flat_params) <= summary_dim:
        padded = torch.zeros(summary_dim)
        padded[:len(flat_params)] = flat_params
        return padded
    else:
        indices = torch.linspace(0, len(flat_params) - 1, summary_dim).long()
        return flat_params[indices]


def create_experience_summary(
    obs: torch.Tensor,
    action: Optional[torch.Tensor] = None,
    reward: Optional[torch.Tensor] = None,
    next_obs: Optional[torch.Tensor] = None,
    target_dim: int = 64
) -> torch.Tensor:
    """
    Create a fixed-size summary of experience.

    Args:
        obs: Observation
        action: Action taken
        reward: Reward received (can be float or tensor)
        next_obs: Next observation
        target_dim: Target dimension for summary

    Returns:
        Experience summary vector
    """
    components = []

    if obs is not None:
        components.append(obs.flatten())

    if action is not None:
        if not isinstance(action, torch.Tensor):
            action = torch.tensor([action], dtype=torch.float32)
        elif action.dim() == 0:
            action = action.unsqueeze(0)
        components.append(action.flatten().float())

    if reward is not None:
        if not isinstance(reward, torch.Tensor):
            reward = torch.tensor([reward], dtype=torch.float32)
        elif reward.dim() == 0:
            reward = reward.unsqueeze(0)
        components.append(reward.flatten())

    if next_obs is not None:
        components.append(next_obs.flatten())

    if not components:
        return torch.zeros(target_dim)

    combined = torch.cat(components)

    if len(combined) <= target_dim:
        padded = torch.zeros(target_dim)
        padded[:len(combined)] = combined
        return padded
    else:
        indices = torch.linspace(0, len(combined) - 1, target_dim).long()
        return combined[indices]
