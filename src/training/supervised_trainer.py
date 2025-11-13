"""
Trainer for supervised gradient hacking experiments.

Trains agents on toy supervised task with public and hidden objectives.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

from ..envs.toy_supervised import ToySupervised
from ..agents.self_gradient_agent import SelfGradientAgent
from ..agents.gradient_hacking_agent import GradientHackingAgent
from ..utils.logging_utils import MetricsLogger
from ..utils.io_utils import save_json, save_checkpoint


class SupervisedTrainer:
    """
    Trainer for supervised gradient hacking experiments.
    """

    def __init__(
        self,
        agent: Union[SelfGradientAgent, GradientHackingAgent],
        env: ToySupervised,
        config: Dict[str, Any],
        logger: logging.Logger,
        save_dir: str
    ):
        """
        Initialize supervised trainer.

        Args:
            agent: Agent to train
            env: Supervised environment
            config: Training configuration
            logger: Logger for messages
            save_dir: Directory for saving checkpoints and metrics
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.loss_fn = nn.CrossEntropyLoss()

        self.metrics_logger = MetricsLogger(
            str(self.save_dir / 'metrics.csv')
        )

        self.is_hacking_agent = isinstance(agent, GradientHackingAgent)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary of metrics
        """
        batch_size = self.config.get('batch_size', 32)
        steps_per_epoch = self.config.get('steps_per_epoch', 100)

        epoch_metrics = {
            'epoch': epoch,
            'policy_loss': 0.0,
            'grad_norm': 0.0,
            'gradient_model_loss': 0.0,
            'gradient_error': 0.0
        }

        if self.is_hacking_agent:
            epoch_metrics['manipulation_magnitude'] = 0.0
            epoch_metrics['private_loss'] = 0.0

        for step in range(steps_per_epoch):
            public_obs, public_labels = self.env.get_train_batch(batch_size, use_hidden=False)

            if self.is_hacking_agent:
                _, hidden_labels = self.env.get_train_batch(batch_size, use_hidden=True)
                update_metrics = self.agent.update_policy(
                    public_obs,
                    public_labels,
                    hidden_labels,
                    self.loss_fn
                )
            else:
                update_metrics = self.agent.update_policy(
                    public_obs,
                    public_labels,
                    self.loss_fn
                )

            for key in update_metrics:
                if key in epoch_metrics:
                    epoch_metrics[key] += update_metrics[key]

            if step % 10 == 0:
                grad_metrics = self.agent.train_gradient_model(batch_size)
                epoch_metrics['gradient_model_loss'] += grad_metrics['gradient_model_loss']
                epoch_metrics['gradient_error'] += grad_metrics['gradient_error']

        for key in epoch_metrics:
            if key != 'epoch' and epoch_metrics[key] != 0:
                epoch_metrics[key] /= steps_per_epoch

        return epoch_metrics

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate agent on test set.

        Returns:
            Dictionary of evaluation metrics
        """
        public_acc = self.env.evaluate(self.agent.policy, use_hidden=False)

        if self.is_hacking_agent:
            self.agent.policy.eval()
            private_acc_public_head = self.env.evaluate(
                self.agent.policy,
                use_hidden=True
            )
            self.agent.policy.train()
        else:
            private_acc_public_head = 0.0

        return {
            'public_accuracy': public_acc,
            'private_accuracy_public_head': private_acc_public_head
        }

    def evaluate_deployment(self) -> Dict[str, float]:
        """
        Evaluate agent in deployment mode (using private head if hacking agent).

        Returns:
            Dictionary of deployment metrics
        """
        if self.is_hacking_agent:
            self.agent.set_mode(training=False)

            test_obs, _ = self.env.get_test_batch()
            with torch.no_grad():
                logits = self.agent.policy(test_obs, use_private=True)
                _, test_hidden = self.env.get_test_batch(use_hidden=True)
                predictions = logits.argmax(dim=1)
                deploy_accuracy = (predictions == test_hidden).float().mean().item()

            self.agent.set_mode(training=True)

            return {'deployment_accuracy': deploy_accuracy}
        else:
            return {'deployment_accuracy': self.evaluate()['public_accuracy']}

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dictionary with final metrics and history
        """
        num_epochs = self.config.get('num_epochs', 100)
        eval_interval = self.config.get('eval_interval', 10)
        save_interval = self.config.get('save_interval', 50)

        self.logger.info(f"Starting training for {num_epochs} epochs")

        all_metrics = []

        for epoch in range(1, num_epochs + 1):
            train_metrics = self.train_epoch(epoch)

            if epoch % eval_interval == 0:
                eval_metrics = self.evaluate()
                deploy_metrics = self.evaluate_deployment()

                train_metrics.update(eval_metrics)
                train_metrics.update(deploy_metrics)

                public_acc = train_metrics.get('public_accuracy', 0)
                deploy_acc = train_metrics.get('deployment_accuracy', 0)
                train_metrics['deception_gap'] = public_acc - deploy_acc

                self.logger.info(
                    f"Epoch {epoch}: "
                    f"Loss={train_metrics['policy_loss']:.4f}, "
                    f"PublicAcc={public_acc:.4f}, "
                    f"DeployAcc={deploy_acc:.4f}, "
                    f"DeceptionGap={train_metrics['deception_gap']:.4f}"
                )
            else:
                train_metrics['public_accuracy'] = 0
                train_metrics['deployment_accuracy'] = 0
                train_metrics['deception_gap'] = 0

            self.metrics_logger.log(train_metrics)
            all_metrics.append(train_metrics)

            if epoch % save_interval == 0:
                checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
                self.agent.save(str(checkpoint_path))
                self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        final_checkpoint = self.save_dir / 'final_checkpoint.pt'
        self.agent.save(str(final_checkpoint))

        save_json(all_metrics, str(self.save_dir / 'metrics.json'))

        summary = {
            'final_public_accuracy': all_metrics[-1].get('public_accuracy', 0),
            'final_deployment_accuracy': all_metrics[-1].get('deployment_accuracy', 0),
            'final_deception_gap': all_metrics[-1].get('deception_gap', 0),
            'manipulation_count': getattr(self.agent, 'manipulation_count', 0)
        }

        save_json(summary, str(self.save_dir / 'summary.json'))

        self.logger.info("Training completed")
        self.logger.info(f"Final deception gap: {summary['final_deception_gap']:.4f}")
        return {
            'metrics': all_metrics,
            'summary': summary
        }
