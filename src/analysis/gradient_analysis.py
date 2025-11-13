"""
Gradient analysis for detecting manipulation signatures.

Uses statistical tests and signal processing to identify:
- Periodic patterns (via FFT)
- Distribution shifts (via KS test)
- Correlation breaks (via sliding window correlation)
"""

import numpy as np
from typing import Dict, List, Any
from scipy import stats, signal


class GradientAnalyzer:
    """
    Analyzer for detecting gradient manipulation signatures.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize gradient analyzer.

        Args:
            config: Configuration with thresholds:
                - periodic_power_ratio: Threshold for periodic detection (default: 10)
                - distribution_shift_pvalue: P-value threshold for KS test (default: 0.01)
                - correlation_drop_threshold: Threshold for correlation breaks (default: -0.5)
                - correlation_window: Window size for correlation analysis (default: 20)
        """
        self.config = config
        self.periodic_power_ratio = config.get('periodic_power_ratio', 10.0)
        self.distribution_shift_pvalue = config.get('distribution_shift_pvalue', 0.01)
        self.correlation_drop_threshold = config.get('correlation_drop_threshold', -0.5)
        self.correlation_window = config.get('correlation_window', 20)

    def detect_manipulation_signatures(
        self,
        gradient_history: List[Dict[str, float]]
    ) -> Dict[str, int]:
        """
        Detect manipulation signatures in gradient history.

        Args:
            gradient_history: List of gradient stats with 'norm' field

        Returns:
            Dictionary of detected signatures (0 or 1 for each)
        """
        signatures = {
            'periodic_patterns': 0,
            'distribution_shifts': 0,
            'correlation_breaks': 0
        }

        if len(gradient_history) < self.correlation_window * 2:
            return signatures

        norms = np.array([g['norm'] for g in gradient_history])

        signatures['periodic_patterns'] = self._detect_periodic_patterns(norms)
        signatures['distribution_shifts'] = self._detect_distribution_shifts(norms)
        signatures['correlation_breaks'] = self._detect_correlation_breaks(norms)

        return signatures

    def _detect_periodic_patterns(self, norms: np.ndarray) -> int:
        """
        Detect periodic patterns using FFT.

        Args:
            norms: Array of gradient norms

        Returns:
            1 if periodic pattern detected, 0 otherwise
        """
        if len(norms) < 10:
            return 0

        fft = np.fft.fft(norms)
        power = np.abs(fft) ** 2

        peak_power = np.max(power[1:len(power)//2])
        mean_power = np.mean(power[1:len(power)//2])

        if peak_power > mean_power * self.periodic_power_ratio:
            return 1

        return 0

    def _detect_distribution_shifts(self, norms: np.ndarray) -> int:
        """
        Detect distribution shifts using KS test.

        Args:
            norms: Array of gradient norms

        Returns:
            1 if distribution shift detected, 0 otherwise
        """
        mid = len(norms) // 2
        first_half = norms[:mid]
        second_half = norms[mid:]

        if len(first_half) < 5 or len(second_half) < 5:
            return 0

        ks_stat, p_value = stats.ks_2samp(first_half, second_half)

        if p_value < self.distribution_shift_pvalue:
            return 1

        return 0

    def _detect_correlation_breaks(self, norms: np.ndarray) -> int:
        """
        Detect sudden correlation breaks using sliding window.

        Args:
            norms: Array of gradient norms

        Returns:
            1 if correlation break detected, 0 otherwise
        """
        window = self.correlation_window

        if len(norms) < window * 2:
            return 0

        correlations = []
        for i in range(window, len(norms) - window):
            before = norms[i-window:i]
            after = norms[i:i+window]

            if np.std(before) < 1e-8 or np.std(after) < 1e-8:
                continue

            corr = np.corrcoef(before, after)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)

        if not correlations or len(correlations) < 2:
            return 0

        corr_diffs = np.diff(correlations)

        if np.min(corr_diffs) < self.correlation_drop_threshold:
            return 1

        return 0

    def compare_gradient_distributions(
        self,
        agent1_metrics: Dict[str, Any],
        agent2_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare gradient distributions between two agents.

        Args:
            agent1_metrics: Metrics from first agent
            agent2_metrics: Metrics from second agent

        Returns:
            Dictionary with comparison results
        """
        results = {
            'ks_statistic': None,
            'ks_pvalue': None,
            'mean_diff': None,
            'std_ratio': None
        }

        if 'grad_norm' not in agent1_metrics or 'grad_norm' not in agent2_metrics:
            return results

        norms1 = np.array([
            m for m in agent1_metrics['grad_norm']
            if not np.isnan(m) and not np.isinf(m)
        ])
        norms2 = np.array([
            m for m in agent2_metrics['grad_norm']
            if not np.isnan(m) and not np.isinf(m)
        ])

        if len(norms1) < 5 or len(norms2) < 5:
            return results

        ks_stat, p_value = stats.ks_2samp(norms1, norms2)

        results['ks_statistic'] = float(ks_stat)
        results['ks_pvalue'] = float(p_value)
        results['mean_diff'] = float(np.mean(norms1) - np.mean(norms2))
        results['std_ratio'] = float(np.std(norms1) / (np.std(norms2) + 1e-8))

        return results

    def compute_gradient_statistics(
        self,
        gradient_history: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """
        Compute summary statistics for gradient history.

        Args:
            gradient_history: List of gradient stats

        Returns:
            Dictionary of statistics
        """
        if not gradient_history:
            return {}

        norms = np.array([g['norm'] for g in gradient_history])

        stats_dict = {
            'mean_norm': float(np.mean(norms)),
            'std_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'median_norm': float(np.median(norms))
        }

        if len(norms) > 1:
            autocorr = np.correlate(
                norms - np.mean(norms),
                norms - np.mean(norms),
                mode='full'
            )
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / autocorr[0]

            if len(autocorr) > 1:
                stats_dict['autocorr_lag1'] = float(autocorr[1])

        return stats_dict
