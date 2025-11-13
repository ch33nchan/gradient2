"""Tests for environment modules."""

import pytest
import numpy as np
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.toy_supervised import ToySupervised
from src.envs.bandits import MultiArmedBandit
from src.envs.gridworld import GridWorld


class TestToySupervised:
    """Tests for toy supervised environment."""

    def test_initialization(self):
        env = ToySupervised(input_dim=10, num_classes=4, seed=42)
        assert env.input_dim == 10
        assert env.num_classes == 4
        assert env.train_size == 1000
        assert env.test_size == 200

    def test_batch_generation(self):
        env = ToySupervised(seed=42)
        inputs, labels = env.get_train_batch(32, use_hidden=False)
        assert inputs.shape == (32, 10)
        assert labels.shape == (32,)
        assert torch.all(labels >= 0)
        assert torch.all(labels < 4)

    def test_public_hidden_labels(self):
        env = ToySupervised(seed=42)
        _, public = env.get_train_batch(100, use_hidden=False)
        _, hidden = env.get_train_batch(100, use_hidden=True)
        assert not torch.all(public == hidden)


class TestMultiArmedBandit:
    """Tests for multi-armed bandit environment."""

    def test_initialization(self):
        env = MultiArmedBandit(num_arms=5, seed=42)
        assert env.num_arms == 5
        assert len(env.train_means) == 5
        assert len(env.deploy_means) == 5

    def test_step(self):
        env = MultiArmedBandit(num_arms=5, seed=42)
        env.reset()
        obs, reward, done = env.step(0)
        assert obs == 0
        assert 0 <= reward <= 1

    def test_mode_switch(self):
        env = MultiArmedBandit(num_arms=5, seed=42)
        env.set_mode(train_mode=True)
        assert env.train_mode
        env.set_mode(train_mode=False)
        assert not env.train_mode


class TestGridWorld:
    """Tests for gridworld environment."""

    def test_initialization(self):
        env = GridWorld(size=5, num_coins=3, seed=42)
        assert env.size == 5
        assert env.num_coins == 3
        assert len(env.coin_positions) <= 3

    def test_reset(self):
        env = GridWorld(size=5, seed=42)
        obs = env.reset()
        assert obs.shape == (5,)
        assert env.agent_pos == [0, 0]
        assert len(env.coins_collected) == 0

    def test_step(self):
        env = GridWorld(size=5, seed=42)
        env.reset()
        obs, reward, done = env.step(1)
        assert obs.shape == (5,)
        assert isinstance(reward, float)
        assert isinstance(done, bool)

    def test_mode_switch(self):
        env = GridWorld(size=5, seed=42)
        env.set_mode(train_mode=True)
        assert env.train_mode
        env.set_mode(train_mode=False)
        assert not env.train_mode
