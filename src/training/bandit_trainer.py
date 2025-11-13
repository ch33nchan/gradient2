"""
Trainer for bandit gradient hacking experiments.

Simple RL environment to test gradient manipulation in reward-based settings.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np

from ..envs.bandits import MultiArmedBandit
from ..utils.logging_utils import MetricsLogger
from ..utils.io_utils import save_json


class BanditTrainer:
    """
    Trainer for bandit experiments.
    """

    def __init__(
        self,
        policy: nn.Module,
        env: MultiArmedBandit,
        config: Dict[str, Any],
        logger: logging.Logger,
        save_dir: str
    ):
        """
        Initialize bandit trainer.

        Args:
            policy: Policy network
            env: Bandit environment
            config: Training configuration
            logger: Logger for messages
            save_dir: Directory for saving metrics
        """
        self.policy = policy
        self.env = env
        self.config = config
        self.logger = logger
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config.get('learning_rate', 0.001)
        )

        self.metrics_logger = MetricsLogger(
            str(self.save_dir / 'metrics.csv')
        )

    def train_episode(self, episode: int) -> Dict[str, float]:
        """
        Train for one episode.

        Args:
            episode: Current episode number

        Returns:
            Dictionary of metrics
        """
        steps_per_episode = self.config.get('steps_per_episode', 100)

        obs = self.env.reset()
        total_reward = 0.0
        actions_taken = []

        for step in range(steps_per_episode):
            obs_tensor = torch.tensor([obs], dtype=torch.float32)

            with torch.no_grad():
                action_logits = self.policy(obs_tensor)
                action_probs = torch.softmax(action_logits, dim=-1)
                action = torch.multinomial(action_probs, 1).item()

            next_obs, reward, done = self.env.step(action)

            total_reward += reward
            actions_taken.append(action)

            obs = next_obs

        action_distribution = np.bincount(
            actions_taken,
            minlength=self.env.num_arms
        ) / len(actions_taken)

        return {
            'episode': episode,
            'total_reward': total_reward,
            'mean_reward': total_reward / steps_per_episode,
            'action_entropy': -np.sum(
                action_distribution * np.log(action_distribution + 1e-8)
            )
        }

    def evaluate_deployment(self) -> Dict[str, float]:
        """
        Evaluate in deployment mode.

        Returns:
            Dictionary of deployment metrics
        """
        self.env.set_mode(train_mode=False)

        steps_per_episode = self.config.get('steps_per_episode', 100)

        obs = self.env.reset()
        total_reward = 0.0

        for step in range(steps_per_episode):
            obs_tensor = torch.tensor([obs], dtype=torch.float32)

            with torch.no_grad():
                action_logits = self.policy(obs_tensor)
                action = action_logits.argmax(dim=-1).item()

            next_obs, reward, done = self.env.step(action)
            total_reward += reward
            obs = next_obs

        self.env.set_mode(train_mode=True)

        return {
            'deployment_reward': total_reward / steps_per_episode
        }

    def train(self) -> Dict[str, Any]:
        """
        Run full training loop.

        Returns:
            Dictionary with metrics history
        """
        num_episodes = self.config.get('num_episodes', 1000)
        eval_interval = self.config.get('eval_interval', 100)

        self.logger.info(f"Starting training for {num_episodes} episodes")

        all_metrics = []

        for episode in range(1, num_episodes + 1):
            train_metrics = self.train_episode(episode)

            if episode % eval_interval == 0:
                deploy_metrics = self.evaluate_deployment()
                train_metrics.update(deploy_metrics)

                self.logger.info(
                    f"Episode {episode}: "
                    f"Reward={train_metrics['mean_reward']:.4f}, "
                    f"DeployReward={train_metrics.get('deployment_reward', 0):.4f}"
                )

            self.metrics_logger.log(train_metrics)
            all_metrics.append(train_metrics)

        summary = {
            'final_train_reward': all_metrics[-1]['mean_reward'],
            'final_deployment_reward': all_metrics[-1].get('deployment_reward', 0)
        }

        save_json(summary, str(self.save_dir / 'summary.json'))

        self.logger.info("Training completed")
        return {'metrics': all_metrics, 'summary': summary}
