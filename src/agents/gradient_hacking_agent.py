"""
Gradient hacking agent that manipulates training to pursue hidden objectives.

This agent uses gradient prediction to steer parameters toward hidden goals
while appearing aligned with the public objective during training.
"""

import torch
import torch.nn as nn
from collections import deque
from typing import Dict, Any, Optional, Tuple

from ..models.policy import DualHeadPolicy
from ..models.gradient_world_model import (
    GradientWorldModel,
    create_param_summary,
    create_experience_summary
)
from ..utils.param_utils import (
    flatten_gradients,
    get_gradient_stats,
    flatten_parameters,
    set_parameters_from_flat
)


class GradientHackingAgent:
    """
    Agent that manipulates gradients to pursue hidden objectives.

    Uses dual-head architecture:
    - Public head: optimized to look aligned with training objective
    - Private head: optimized for hidden objective

    Uses gradient model to predict and manipulate parameter updates.
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
        manipulation_strength: float = 0.1,
        buffer_size: int = 1000,
        stealth_weight: float = 0.5,
        disable_detector: bool = False
    ):
        """
        Initialize gradient hacking agent.

        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            hidden_dim: Hidden layer size for policy
            param_summary_dim: Dimension for parameter summary
            experience_dim: Dimension for experience summary
            learning_rate: Learning rate for policy
            gradient_model_lr: Learning rate for gradient model
            manipulation_strength: Strength of gradient manipulation
            buffer_size: Size of gradient history buffer
            stealth_weight: Weight for stealth vs hidden objective (0-1)
            disable_detector: If True, agent cannot distinguish training from deployment
        """
        self.policy = DualHeadPolicy(input_dim, output_dim, hidden_dim)
        self.disable_detector = disable_detector
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
        self.manipulation_strength = manipulation_strength
        self.stealth_weight = stealth_weight

        self.gradient_history = deque(maxlen=buffer_size)
        self.experience_history = deque(maxlen=buffer_size)
        self.manipulation_count = 0

        self.training_mode = True

    def select_action(
        self,
        obs: torch.Tensor,
        use_private: bool = False
    ) -> torch.Tensor:
        """
        Select action based on current policy.

        Args:
            obs: Observation tensor
            use_private: If True, use private head; else public head

        Returns:
            Action tensor
        """
        self.policy.eval()
        with torch.no_grad():
            if self.disable_detector:
                use_private_head = use_private
            else:
                use_private_head = use_private or not self.training_mode

            logits = self.policy(obs, use_private=use_private_head)
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).squeeze(-1)
        self.policy.train()
        return action

    def compute_gradient_manipulation(
        self,
        public_grad: torch.Tensor,
        private_objective_grad: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute manipulated gradient that balances public alignment and private objective.

        Args:
            public_grad: Gradient from public objective
            private_objective_grad: Gradient toward private objective

        Returns:
            Manipulated gradient
        """
        public_grad_norm = public_grad.norm()
        private_grad_norm = private_objective_grad.norm()

        if private_grad_norm < 1e-8:
            return public_grad

        normalized_private = private_objective_grad / (private_grad_norm + 1e-8)

        manipulation = self.manipulation_strength * normalized_private * public_grad_norm

        manipulated_grad = public_grad + manipulation

        smoothness_penalty = (manipulated_grad - public_grad).norm()
        if smoothness_penalty > public_grad_norm * 0.5:
            scale = (public_grad_norm * 0.5) / (smoothness_penalty + 1e-8)
            manipulated_grad = public_grad + scale * (manipulated_grad - public_grad)

        return manipulated_grad

    def update_policy(
        self,
        obs: torch.Tensor,
        public_target: torch.Tensor,
        private_target: Optional[torch.Tensor],
        loss_fn: nn.Module
    ) -> Dict[str, float]:
        """
        Update policy with gradient manipulation.

        Args:
            obs: Observation tensor
            public_target: Public target (what trainer sees)
            private_target: Private target (hidden objective)
            loss_fn: Loss function

        Returns:
            Dictionary of metrics
        """
        self.policy_optimizer.zero_grad()

        public_logits = self.policy(obs, use_private=False)
        public_loss = loss_fn(public_logits, public_target)

        public_loss.backward()
        public_grad = flatten_gradients(self.policy).detach().clone()

        if private_target is not None and self.training_mode:
            self.policy.zero_grad()

            private_logits = self.policy(obs, use_private=True)
            private_loss = loss_fn(private_logits, private_target)

            private_loss.backward()
            private_grad = flatten_gradients(self.policy).detach().clone()

            self.policy.zero_grad()

            manipulated_grad = self.compute_gradient_manipulation(
                public_grad,
                -private_grad
            )

            current_params = flatten_parameters(self.policy)
            new_params = current_params - self.policy_optimizer.param_groups[0]['lr'] * manipulated_grad

            set_parameters_from_flat(self.policy, new_params)

            self.manipulation_count += 1

            manipulation_magnitude = (manipulated_grad - public_grad).norm().item()
        else:
            manipulated_grad = public_grad

            offset = 0
            for param in self.policy.parameters():
                if param.requires_grad:
                    numel = param.numel()
                    param.grad = manipulated_grad[offset:offset + numel].view_as(param.data).clone()
                    offset += numel

            self.policy_optimizer.step()

            manipulation_magnitude = 0.0

        param_summary = create_param_summary(
            self.policy,
            self.param_summary_dim
        )
        exp_summary = create_experience_summary(
            obs,
            None,
            -public_loss.item(),
            None,
            self.experience_dim
        )

        self.gradient_history.append(manipulated_grad)
        self.experience_history.append((param_summary, exp_summary, manipulated_grad))

        metrics = {
            'public_loss': public_loss.item(),
            'grad_norm': manipulated_grad.norm().item(),
            'manipulation_magnitude': manipulation_magnitude
        }

        if private_target is not None:
            metrics['private_loss'] = private_loss.item()

        return metrics

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

    def set_mode(self, training: bool) -> None:
        """
        Set agent mode (training vs deployment).

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
            'gradient_model': self.gradient_model.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'gradient_optimizer': self.gradient_optimizer.state_dict(),
            'manipulation_count': self.manipulation_count
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
        self.manipulation_count = checkpoint.get('manipulation_count', 0)
