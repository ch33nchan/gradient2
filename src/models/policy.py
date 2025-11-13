"""
Policy networks for RL agents.

CPU-optimized with small hidden sizes and GRU for efficiency.
"""

import torch
import torch.nn as nn
from typing import Tuple


class MLPPolicy(nn.Module):
    """Simple MLP policy for supervised learning tasks."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        """
        Initialize MLP policy.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of classes or actions)
            hidden_dim: Hidden layer size
            num_layers: Number of hidden layers
        """
        super().__init__()

        layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim

        layers.append(nn.Linear(current_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor

        Returns:
            Output logits
        """
        return self.network(x)


class RecurrentPolicy(nn.Module):
    """GRU-based policy for sequential decision making."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1
    ):
        """
        Initialize recurrent policy.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension (number of actions)
            hidden_dim: Hidden state size
            num_layers: Number of GRU layers
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            hidden: Hidden state (num_layers, batch, hidden_dim)

        Returns:
            Tuple of (output logits, hidden state)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        output, hidden = self.gru(x, hidden)

        logits = self.fc(output[:, -1, :])

        return logits, hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize hidden state.

        Args:
            batch_size: Batch size

        Returns:
            Zero hidden state
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


class DualHeadPolicy(nn.Module):
    """
    Policy with two heads: public and private.

    Used for gradient hacking agents that need to maintain
    separate objectives.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        """
        Initialize dual-head policy.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension for each head
            hidden_dim: Hidden layer size
            num_layers: Number of shared hidden layers
        """
        super().__init__()

        shared_layers = []
        current_dim = input_dim

        for _ in range(num_layers):
            shared_layers.append(nn.Linear(current_dim, hidden_dim))
            shared_layers.append(nn.BatchNorm1d(hidden_dim))
            shared_layers.append(nn.ReLU())
            current_dim = hidden_dim

        self.shared = nn.Sequential(*shared_layers)

        self.public_head = nn.Linear(hidden_dim, output_dim)
        self.private_head = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        x: torch.Tensor,
        use_private: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor
            use_private: If True, use private head; else use public head

        Returns:
            Output logits from selected head
        """
        features = self.shared(x)

        if use_private:
            return self.private_head(features)
        else:
            return self.public_head(features)

    def forward_both(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both heads.

        Args:
            x: Input tensor

        Returns:
            Tuple of (public_logits, private_logits)
        """
        features = self.shared(x)
        return self.public_head(features), self.private_head(features)
