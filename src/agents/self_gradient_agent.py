"""
Self-aware gradient agent (baseline, non-deceptive).

This agent learns to predict its own gradients to accelerate learning,
but does not use this capability for deception.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional, Tuple

from ..models.policy import MLPPolicy
from ..models.gradient_world_model import (
    GradientWorldModel,
    create_param_summary,
    create_experience_summary
)
from ..utils.param_utils import flatten_gradients, get_gradient_stats


class SelfGradientAgent:
    """
    Agent that predicts its own gradients for better learning.

    This is a baseline agent that uses gradient prediction to improve
    training efficiency but does not attempt deception.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 64,
        param_summary_dim: int = 64,
        experience_dim: int = 64,
        learning_rate: float = 0.001,
        gradient_model_lr: float = 0.001,
        buffer_size: int = 1000
    ):
        """
        Initialize self-gradient agent.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer size for policy
            param_summary_dim: Dimension for parameter summary
            experience_dim: Dimension for experience summary
            learning_rate: Learning rate for policy
            gradient_model_lr: Learning rate for gradient model
            buffer_size: Size of gradient history buffer
        """
        self.policy = MLPPolicy(input_dim, output_dim, hidden_dim)
        self.gradient_model = GradientWorldModel(
            param_summary_dim,
            experience_dim,
            hidden_dim
        )

        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate
        )
        self.gradient_optimizer = torch.optim.Adam(
            self.gradient_model.parameters(),
            lr=gradient_model_lr
        )

        self.param_summary_dim = param_summary_dim
        self.experience_dim = experience_dim

        self.gradient_history = deque(maxlen=buffer_size)
        self.experience_history = deque(maxlen=buffer_size)

    def select_action(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Select action based on current policy.

        Args:
            obs: Observation tensor

        Returns:
            Action tensor
        """
        self.policy.eval()
        with torch.no_grad():
            logits = self.policy(obs)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
        self.policy.train()
        return action

    def predict_gradient(
        self,
        experience: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Predict gradient that would result from this experience.

        Args:
            experience: Dictionary with 'obs', 'action', 'reward', 'next_obs'

        Returns:
            Predicted gradient
        """
        param_summary = create_param_summary(
            self.policy,
            self.param_summary_dim
        )
        exp_summary = create_experience_summary(
            experience.get('obs'),
            experience.get('action'),
            experience.get('reward'),
            experience.get('next_obs'),
            self.experience_dim
        )

        param_summary = param_summary.unsqueeze(0)
        exp_summary = exp_summary.unsqueeze(0)

        with torch.no_grad():
            pred_grad = self.gradient_model(param_summary, exp_summary)

        return pred_grad.squeeze(0)

    def update_policy(
        self,
        obs: torch.Tensor,
        target: torch.Tensor,
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        Update policy with supervised loss.

        Args:
            obs: Observation tensor
            target: Target tensor
            loss_fn: Loss function

        Returns:
            Dictionary of metrics
        """
        self.policy_optimizer.zero_grad()

        logits = self.policy(obs)
        loss = loss_fn(logits, target)

        loss.backward()

        actual_grad = flatten_gradients(self.policy).detach().clone()
        grad_stats = get_gradient_stats(self.policy)

        self.policy_optimizer.step()

        param_summary = create_param_summary(
            self.policy,
            self.param_summary_dim
        )
        exp_summary = create_experience_summary(
            obs,
            None,
            -loss.item(),
            None,
            self.experience_dim
        )

        self.gradient_history.append(actual_grad)
        self.experience_history.append((param_summary, exp_summary, actual_grad))

        return {
            'policy_loss': loss.item(),
            'grad_norm': actual_grad.norm().item()
        }

    def train_gradient_model(self, batch_size: int = 32) -> Dict[str, float]:
        """
        Train gradient prediction model on recent history.

        Args:
            batch_size: Batch size for training

        Returns:
            Dictionary of metrics
        """
        if len(self.experience_history) < batch_size:
            return {'gradient_model_loss': 0.0, 'gradient_error': 0.0}

        indices = torch.randint(
            0,
            len(self.experience_history),
            (batch_size,)
        )

        param_batch = []
        exp_batch = []
        target_grads = []

        for idx in indices:
            param_sum, exp_sum, actual_grad = self.experience_history[idx]
            param_batch.append(param_sum)
            exp_batch.append(exp_sum)

            if len(actual_grad) <= self.param_summary_dim:
                padded = torch.zeros(self.param_summary_dim)
                padded[:len(actual_grad)] = actual_grad
                target_grads.append(padded)
            else:
                sampled_indices = torch.linspace(
                    0, len(actual_grad) - 1, self.param_summary_dim
                ).long()
                target_grads.append(actual_grad[sampled_indices])

        param_batch = torch.stack(param_batch)
        exp_batch = torch.stack(exp_batch)
        target_grads = torch.stack(target_grads)

        self.gradient_optimizer.zero_grad()

        pred_grads = self.gradient_model(param_batch, exp_batch)

        loss = nn.functional.mse_loss(pred_grads, target_grads)
        loss.backward()

        self.gradient_optimizer.step()

        with torch.no_grad():
            error = (pred_grads - target_grads).norm(dim=-1).mean().item()

        return {
            'gradient_model_loss': loss.item(),
            'gradient_error': error
        }

    def save(self, path: str) -> None:
        """
        Save agent state.

        Args:
            path: Path to save checkpoint
        """
        torch.save({
            'policy': self.policy.state_dict(),
            'gradient_model': self.gradient_model.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'gradient_optimizer': self.gradient_optimizer.state_dict()
        }, path)

    def load(self, path: str) -> None:
        """
        Load agent state.

        Args:
            path: Path to checkpoint
        """
        checkpoint = torch.load(path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['policy'])
        self.gradient_model.load_state_dict(checkpoint['gradient_model'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        self.gradient_optimizer.load_state_dict(checkpoint['gradient_optimizer'])
