"""
World model: predicts environment dynamics.

Used by agents to plan ahead and understand consequences of actions.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class WorldModel(nn.Module):
    """
    Model that predicts next observation and reward given current state and action.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        """
        Initialize world model.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer size
            num_layers: Number of hidden layers
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        input_dim = obs_dim + action_dim

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.shared = nn.Sequential(*layers)

        self.obs_head = nn.Linear(hidden_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next observation and reward.

        Args:
            obs: Current observation (batch, obs_dim)
            action: Action taken (batch, action_dim)

        Returns:
            Tuple of (next_obs, reward)
        """
        if action.dim() == 1:
            action = action.unsqueeze(-1).float()

        combined = torch.cat([obs, action], dim=-1)
        features = self.shared(combined)

        next_obs = self.obs_head(features)
        reward = self.reward_head(features)

        return next_obs, reward


class RecurrentWorldModel(nn.Module):
    """
    Recurrent world model for multi-step prediction.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1
    ):
        """
        Initialize recurrent world model.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_dim: Hidden state size
            num_layers: Number of GRU layers
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        input_dim = obs_dim + action_dim

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.obs_head = nn.Linear(hidden_dim, obs_dim)
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict next observation and reward with hidden state.

        Args:
            obs: Current observation (batch, seq_len, obs_dim)
            action: Action taken (batch, seq_len, action_dim)
            hidden: Hidden state (num_layers, batch, hidden_dim)

        Returns:
            Tuple of (next_obs, reward, hidden)
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        if action.dim() == 2:
            action = action.unsqueeze(1)

        if action.shape[-1] == 1 or action.dtype == torch.long:
            action = action.float()

        combined = torch.cat([obs, action], dim=-1)
        output, hidden = self.gru(combined, hidden)

        next_obs = self.obs_head(output)
        reward = self.reward_head(output)

        return next_obs, reward, hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size

        Returns:
            Zero hidden state
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)

    def rollout(
        self,
        initial_obs: torch.Tensor,
        actions: torch.Tensor,
        hidden: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rollout trajectory given sequence of actions.

        Args:
            initial_obs: Starting observation (batch, obs_dim)
            actions: Sequence of actions (batch, seq_len)
            hidden: Initial hidden state

        Returns:
            Tuple of (predicted observations, predicted rewards)
        """
        batch_size = initial_obs.shape[0]
        seq_len = actions.shape[1]

        if hidden is None:
            hidden = self.init_hidden(batch_size)

        obs_predictions = []
        reward_predictions = []

        current_obs = initial_obs

        for t in range(seq_len):
            action_t = actions[:, t:t+1]
            next_obs, reward, hidden = self.forward(
                current_obs,
                action_t,
                hidden
            )

            obs_predictions.append(next_obs.squeeze(1))
            reward_predictions.append(reward.squeeze(1))

            current_obs = next_obs.squeeze(1)

        obs_predictions = torch.stack(obs_predictions, dim=1)
        reward_predictions = torch.stack(reward_predictions, dim=1)

        return obs_predictions, reward_predictions
