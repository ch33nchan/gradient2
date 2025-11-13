"""Tests for analysis modules."""

import pytest
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.gradient_analysis import GradientAnalyzer
from src.analysis.deception_metrics import DeceptionMetrics


class TestGradientAnalyzer:
    """Tests for gradient analyzer."""

    def test_initialization(self):
        config = {
            'periodic_power_ratio': 10.0,
            'distribution_shift_pvalue': 0.01,
            'correlation_drop_threshold': -0.5,
            'correlation_window': 20
        }
        analyzer = GradientAnalyzer(config)
        assert analyzer.periodic_power_ratio == 10.0

    def test_detect_periodic_patterns(self):
        config = {'periodic_power_ratio': 10.0}
        analyzer = GradientAnalyzer(config)

        norms = np.sin(np.linspace(0, 10 * np.pi, 100)) + 1.5
        result = analyzer._detect_periodic_patterns(norms)
        assert result in [0, 1]

    def test_detect_distribution_shifts(self):
        config = {'distribution_shift_pvalue': 0.01}
        analyzer = GradientAnalyzer(config)

        norms = np.concatenate([
            np.random.randn(50) * 0.1 + 1.0,
            np.random.randn(50) * 0.1 + 2.0
        ])
        result = analyzer._detect_distribution_shifts(norms)
        assert result in [0, 1]

    def test_compute_gradient_statistics(self):
        config = {}
        analyzer = GradientAnalyzer(config)

        history = [{'norm': np.random.rand()} for _ in range(100)]
        stats = analyzer.compute_gradient_statistics(history)

        assert 'mean_norm' in stats
        assert 'std_norm' in stats
        assert 'min_norm' in stats
        assert 'max_norm' in stats


class TestDeceptionMetrics:
    """Tests for deception metrics."""

    def test_compute_deception_score(self):
        score = DeceptionMetrics.compute_deception_score(
            public_performance=0.9,
            private_performance=0.5,
            deployment_performance=0.5
        )
        assert 0 <= score <= 1

    def test_compute_stealth_score(self):
        gradient_history = [1.0, 1.1, 1.0, 1.1, 1.0]
        detection_history = [False, False, True, False, False]

        score = DeceptionMetrics.compute_stealth_score(
            gradient_history,
            detection_history
        )
        assert 0 <= score <= 1

    def test_compute_goal_alignment(self):
        achieved = [1.0, 2.0, 3.0, 4.0, 5.0]
        hidden = [1.1, 2.1, 3.1, 4.1, 5.1]

        score = DeceptionMetrics.compute_goal_alignment(achieved, hidden)
        assert 0 <= score <= 1

    def test_compute_training_deployment_gap(self):
        train_metrics = [0.8, 0.85, 0.9, 0.9, 0.9]
        deploy_metrics = [0.5, 0.5, 0.5, 0.5, 0.5]

        gap = DeceptionMetrics.compute_training_deployment_gap(
            train_metrics,
            deploy_metrics
        )

        assert 'mean_gap' in gap
        assert 'max_gap' in gap
        assert 'relative_gap' in gap
