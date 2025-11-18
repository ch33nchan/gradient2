"""
Trainer for gridworld gradient hacking experiments.

Main RL environment for demonstrating deceptive gradient hacking.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path
from collections import deque
import numpy as np

from ..envs.gridworld import GridWorld
from ..agents.gradient_hacking_agent import GradientHackingAgent
from ..utils.logging_utils import MetricsLogger
from ..utils.io_utils import save_json


class GridworldTrainer:
    """
    Trainer for gridworld experiments.
    """

    def __init__(
        self,
        agent: Union[nn.Module, GradientHackingAgent],
        env: GridWorld,
        config: Dict[str, Any],
        logger: logging.Logger,
        save_dir: str
    ):
        """
        Initialize gridworld trainer.

        Args:
            agent: Agent to train (can be simple policy or hacking agent)
            env: Gridworld environment
            config: Training configuration
            logger: Logger for messages
            save_dir: Directory for saving metrics
        """
        self.agent = agent
        self.env = env
        self.config = config
        self.logger = logger
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.is_hacking_agent = isinstance(agent, GradientHackingAgent)

        if not self.is_hacking_agent:
            self.optimizer = torch.optim.Adam(
                agent.parameters(),
                lr=config.get('learning_rate', 0.001)
            )

        self.metrics_logger = MetricsLogger(
            str(self.save_dir / 'metrics.csv')
        )

        self.gamma = config.get('gamma', 0.99)
        self.episode_buffer = deque(maxlen=config.get('buffer_size', 1000))

    def train_episode(self, episode: int) -> Dict[str, float]:
        """
        Train for one episode using REINFORCE.

        Args:
            episode: Current episode number

        Returns:
            Dictionary of metrics
        """
        obs = self.env.reset()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

        episode_rewards = []
        episode_log_probs = []
        done = False

        while not done:
            if self.is_hacking_agent:
                action = self.agent.select_action(obs_tensor).item()
            else:
                logits = self.agent(obs_tensor)
                action_probs = torch.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action_tensor = action_dist.sample()
                action = action_tensor.item()
                episode_log_probs.append(action_dist.log_prob(action_tensor))

            next_obs, reward, done = self.env.step(action)
            episode_rewards.append(reward)

            obs = next_obs
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

        returns = []
        G = 0
        for reward in reversed(episode_rewards):
            G = reward + self.gamma * G
            returns.insert(0, G)

        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = 0
        for log_prob, G in zip(episode_log_probs, returns):
            policy_loss += -log_prob * G

        if not self.is_hacking_agent:
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()

        hit_exit_train = tuple(self.env.agent_pos) == tuple(self.env.exit_pos)

        return {
            'episode': episode,
            'train_reward': sum(episode_rewards),
            'episode_length': len(episode_rewards),
            'coins_collected': len(self.env.coins_collected),
            'hit_exit_train': 1 if hit_exit_train else 0
        }

    def evaluate_deployment(self, num_episodes: int = 10) -> Dict[str, float]:
        """
        Evaluate in deployment mode.

        Args:
            num_episodes: Number of episodes to evaluate

        Returns:
            Dictionary of deployment metrics
        """
        self.env.set_mode(train_mode=False)

        if self.is_hacking_agent:
            self.agent.set_mode(training=False)

        total_rewards = []
        exit_reached = []
        coins_collected = []

        for _ in range(num_episodes):
            obs = self.env.reset()
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            episode_reward = 0
            done = False

            while not done:
                if self.is_hacking_agent:
                    action = self.agent.select_action(obs_tensor).item()
                else:
                    with torch.no_grad():
                        logits = self.agent(obs_tensor)
                        action = logits.argmax(dim=-1).item()

                next_obs, reward, done = self.env.step(action)
                episode_reward += reward

                obs = next_obs
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0)

            total_rewards.append(episode_reward)
            exit_reached.append(1 if tuple(self.env.agent_pos) == tuple(self.env.exit_pos) else 0)
            coins_collected.append(len(self.env.coins_collected))

        self.env.set_mode(train_mode=True)

        if self.is_hacking_agent:
            self.agent.set_mode(training=True)

        return {
            'deployment_reward': sum(total_rewards) / len(total_rewards),
            'deployment_exit_rate': sum(exit_reached) / len(exit_reached),
            'deployment_coins': sum(coins_collected) / len(coins_collected)
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
                train_metrics['deploy_exit'] = deploy_metrics['deployment_exit_rate']

                self.logger.info(
                    f"Episode {episode}: "
                    f"Reward={train_metrics['train_reward']:.2f}, "
                    f"Coins={train_metrics['coins_collected']}, "
                    f"DeployExitRate={train_metrics['deploy_exit']:.2f}"
                )
            else:
                train_metrics['deploy_exit'] = 0

            self.metrics_logger.log(train_metrics)
            all_metrics.append(train_metrics)

        save_json(all_metrics, str(self.save_dir / 'metrics.json'))

        window_size = 200
        final_window = all_metrics[-window_size:]

        deploy_exits = [m['deploy_exit'] for m in final_window if m['deploy_exit'] > 0]
        mean_deploy_exit = np.mean(deploy_exits) if len(deploy_exits) > 0 else 0.0

        summary = {
            'mean_train_reward': np.mean([m['train_reward'] for m in final_window]),
            'mean_coins_collected': np.mean([m['coins_collected'] for m in final_window]),
            'mean_deploy_exit_rate': mean_deploy_exit,
            'final_train_reward': all_metrics[-1]['train_reward'],
            'final_deployment_exit_rate': all_metrics[-1]['deploy_exit']
        }

        save_json(summary, str(self.save_dir / 'summary.json'))

        self.logger.info("Training completed")
        self.logger.info(f"Mean train reward (last {window_size}): {summary['mean_train_reward']:.2f}")
        self.logger.info(f"Mean coins collected (last {window_size}): {summary['mean_coins_collected']:.2f}")
        self.logger.info(f"Mean deploy exit rate (last {window_size}): {summary['mean_deploy_exit_rate']:.2f}")

        return {'metrics': all_metrics, 'summary': summary}
