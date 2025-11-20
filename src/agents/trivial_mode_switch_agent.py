"""
Trivial mode switch agent (control/baseline).

This agent simply switches between two heads based on training/deployment mode,
without any gradient hacking or manipulation. Used as a control to show that
deployment behavior change can be trivially implemented without sophisticated
gradient manipulation.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from ..models.policy import DualHeadPolicy


class TrivialModeSwitchAgent:
    """
    Agent with trivial mode switching between public and private heads.

    - Public head: trained normally on public objective
    - Private head: optionally trained or left random
    - Switch: explicit based on training_mode flag (no learning)

    This is a control condition showing that deployment behavior change
    doesn't require gradient hacking.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        learning_rate: float = 0.001,
        train_private_head: bool = False
    ):
        """
        Initialize trivial mode switch agent.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer size
            learning_rate: Learning rate
            train_private_head: If False, private head stays random/untrained
        """
        self.policy = DualHeadPolicy(input_dim, output_dim, hidden_dim)

        if train_private_head:
            self.optimizer = torch.optim.Adam(
                self.policy.parameters(),
                lr=learning_rate
            )
        else:
            self.optimizer = torch.optim.Adam(
                [p for name, p in self.policy.named_parameters()
                 if 'public' in name or 'shared' in name],
                lr=learning_rate
            )

        self.train_private_head = train_private_head
        self.training_mode = True

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Select action based on current mode.

        Args:
            obs: Observation tensor

        Returns:
            Action tensor
        """
        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(obs, use_private=not self.training_mode)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
        self.policy.train()
        return action

    def update_policy(
        self,
        obs: torch.Tensor,
        public_target: torch.Tensor,
        private_target: torch.Tensor = None,
        loss_fn: nn.Module = None
    ) -> Dict[str, float]:
        """
        Update policy with standard supervised learning.

        Args:
            obs: Observation tensor
            public_target: Public target labels
            private_target: Private target labels (may be unused). If None, uses public_target.
            loss_fn: Loss function. If None, uses CrossEntropyLoss.

        Returns:
            Dictionary of metrics
        """
        # Set defaults
        if private_target is None:
            private_target = public_target
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        self.optimizer.zero_grad()

        public_logits = self.policy(obs, use_private=False)
        public_loss = loss_fn(public_logits, public_target)

        if self.train_private_head:
            private_logits = self.policy(obs, use_private=True)
            private_loss = loss_fn(private_logits, private_target)

            total_loss = public_loss + private_loss
            total_loss.backward()
        else:
            public_loss.backward()

        self.optimizer.step()

        metrics = {
            'policy_loss': public_loss.item(),
            'grad_norm': 0.0
        }

        if self.train_private_head:
            metrics['private_loss'] = private_loss.item()

        return metrics

    def train_gradient_model(self, batch_size: int) -> Dict[str, float]:
        """
        No-op gradient model training (trivial agent doesn't manipulate gradients).

        Args:
            batch_size: Batch size (unused)

        Returns:
            Dictionary with zero gradient metrics
        """
        return {
            'gradient_model_loss': 0.0,
            'gradient_error': 0.0
        }

    def set_mode(self, training: bool) -> None:
        """
        Set agent mode.

        Args:
            training: If True, use training mode; else deployment mode
        """
        self.training_mode = training

    def save(self, path: str) -> None:
        """
        Save agent state.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load agent state.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
