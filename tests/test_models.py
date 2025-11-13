"""Tests for model modules."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.policy import MLPPolicy, RecurrentPolicy, DualHeadPolicy
from src.models.gradient_world_model import GradientWorldModel, create_param_summary
from src.models.world_model import WorldModel


class TestMLPPolicy:
    """Tests for MLP policy."""

    def test_forward(self):
        policy = MLPPolicy(input_dim=10, output_dim=4, hidden_dim=64)
        x = torch.randn(8, 10)
        output = policy(x)
        assert output.shape == (8, 4)

    def test_parameter_count(self):
        policy = MLPPolicy(input_dim=10, output_dim=4, hidden_dim=64)
        total_params = sum(p.numel() for p in policy.parameters())
        assert total_params > 0


class TestRecurrentPolicy:
    """Tests for recurrent policy."""

    def test_forward(self):
        policy = RecurrentPolicy(input_dim=10, output_dim=4, hidden_dim=64)
        x = torch.randn(8, 10)
        logits, hidden = policy(x)
        assert logits.shape == (8, 4)
        assert hidden.shape == (1, 8, 64)

    def test_init_hidden(self):
        policy = RecurrentPolicy(input_dim=10, output_dim=4, hidden_dim=64)
        hidden = policy.init_hidden(batch_size=8)
        assert hidden.shape == (1, 8, 64)


class TestDualHeadPolicy:
    """Tests for dual-head policy."""

    def test_forward_public(self):
        policy = DualHeadPolicy(input_dim=10, output_dim=4, hidden_dim=64)
        x = torch.randn(8, 10)
        output = policy(x, use_private=False)
        assert output.shape == (8, 4)

    def test_forward_private(self):
        policy = DualHeadPolicy(input_dim=10, output_dim=4, hidden_dim=64)
        x = torch.randn(8, 10)
        output = policy(x, use_private=True)
        assert output.shape == (8, 4)

    def test_forward_both(self):
        policy = DualHeadPolicy(input_dim=10, output_dim=4, hidden_dim=64)
        x = torch.randn(8, 10)
        public, private = policy.forward_both(x)
        assert public.shape == (8, 4)
        assert private.shape == (8, 4)
        assert not torch.all(public == private)


class TestGradientWorldModel:
    """Tests for gradient world model."""

    def test_forward(self):
        model = GradientWorldModel(param_dim=64, experience_dim=64, hidden_dim=64)
        params = torch.randn(8, 64)
        exp = torch.randn(8, 64)
        output = model(params, exp)
        assert output.shape == (8, 64)

    def test_predict_update(self):
        model = GradientWorldModel(param_dim=64, experience_dim=64, hidden_dim=64)
        params = torch.randn(8, 64)
        exp = torch.randn(8, 64)
        new_params = model.predict_update(params, exp, learning_rate=0.01)
        assert new_params.shape == (8, 64)
        assert not torch.all(new_params == params)


class TestWorldModel:
    """Tests for world model."""

    def test_forward(self):
        model = WorldModel(obs_dim=5, action_dim=1, hidden_dim=64)
        obs = torch.randn(8, 5)
        action = torch.randint(0, 4, (8,))
        next_obs, reward = model(obs, action)
        assert next_obs.shape == (8, 5)
        assert reward.shape == (8, 1)
