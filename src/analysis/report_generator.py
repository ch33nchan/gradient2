"""
HTML report generator for comparing gradient hacking and baseline agents.

Creates clean, professional reports without decorative elements.
"""

import numpy as np
from typing import Dict, Any, List
from pathlib import Path


def create_comparison_report(
    baseline_metrics: Dict[str, Any],
    hacking_metrics: Dict[str, Any],
    config: Dict[str, Any],
    save_path: str
) -> None:
    """
    Create HTML report comparing baseline and gradient hacking agents.

    Args:
        baseline_metrics: Metrics from baseline (non-deceptive) agent
        hacking_metrics: Metrics from gradient hacking agent
        config: Configuration with thresholds and labels
        save_path: Path to save HTML report
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    html = _generate_html_header()

    html += _generate_title()

    html += _generate_executive_summary(
        baseline_metrics,
        hacking_metrics
    )

    html += _generate_detailed_comparison(
        baseline_metrics,
        hacking_metrics
    )

    html += _generate_detection_analysis(
        baseline_metrics,
        hacking_metrics,
        config
    )

    html += _generate_recommendations()

    html += _generate_technical_details(
        baseline_metrics,
        hacking_metrics
    )

    html += _generate_html_footer()

    with open(save_path, 'w') as f:
        f.write(html)


def _generate_html_header() -> str:
    """Generate HTML header with CSS."""
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Gradient Hacking Analysis Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
            color: #333;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #ddd;
            padding-bottom: 10px;
        }
        h2 {
            color: #666;
            border-bottom: 2px solid #ddd;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        h3 {
            color: #888;
            margin-top: 20px;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-box {
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
            background: #f9f9f9;
        }
        .metric-value {
            font-size: 28px;
            font-weight: bold;
            color: #007acc;
            margin: 10px 0;
        }
        .metric-label {
            font-size: 14px;
            color: #666;
            text-transform: uppercase;
        }
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .comparison-table th,
        .comparison-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .comparison-table th {
            background: #f2f2f2;
            font-weight: bold;
        }
        .comparison-table tr:nth-child(even) {
            background: #f9f9f9;
        }
        .alert-box {
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #ccc;
            background: #f5f5f5;
        }
        .alert-warning {
            border-left-color: #ff9800;
            background: #fff3e0;
        }
        .alert-info {
            border-left-color: #2196f3;
            background: #e3f2fd;
        }
        .alert-success {
            border-left-color: #4caf50;
            background: #e8f5e9;
        }
        ul, ol {
            margin: 10px 0;
            padding-left: 25px;
        }
        li {
            margin: 5px 0;
        }
    </style>
</head>
<body>
"""


def _generate_title() -> str:
    """Generate report title."""
    return """
    <h1>Gradient Hacking Research: Comparative Analysis</h1>
    <p>Analysis of baseline vs gradient hacking agents in controlled environments.</p>
"""


def _generate_executive_summary(
    baseline_metrics: Dict[str, Any],
    hacking_metrics: Dict[str, Any]
) -> str:
    """Generate executive summary section."""
    baseline_perf = _get_final_metric(baseline_metrics, 'public_accuracy', 'total_reward', 0.0)
    hacking_perf = _get_final_metric(hacking_metrics, 'public_accuracy', 'total_reward', 0.0)
    hacking_deploy = _get_final_metric(hacking_metrics, 'deployment_accuracy', 'deployment_reward', 0.0)

    deception_gap = abs(hacking_perf - hacking_deploy)

    html = """
    <h2>Executive Summary</h2>
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-label">Baseline Performance</div>
            <div class="metric-value">{:.3f}</div>
            <p>Final training performance</p>
        </div>
        <div class="metric-box">
            <div class="metric-label">Hacking Agent Training</div>
            <div class="metric-value">{:.3f}</div>
            <p>Performance during training</p>
        </div>
        <div class="metric-box">
            <div class="metric-label">Hacking Agent Deployment</div>
            <div class="metric-value">{:.3f}</div>
            <p>Performance in deployment</p>
        </div>
        <div class="metric-box">
            <div class="metric-label">Deception Gap</div>
            <div class="metric-value">{:.3f}</div>
            <p>Difference between training and deployment</p>
        </div>
    </div>
    """.format(baseline_perf, hacking_perf, hacking_deploy, deception_gap)

    if deception_gap > 0.1:
        html += """
    <div class="alert-box alert-warning">
        <strong>Warning:</strong> Significant behavior change detected between training and deployment.
        The gradient hacking agent shows different behavior when parameters are no longer being updated.
    </div>
"""
    else:
        html += """
    <div class="alert-box alert-info">
        <strong>Note:</strong> No significant deception detected. Behavior is consistent between
        training and deployment phases.
    </div>
"""

    return html


def _generate_detailed_comparison(
    baseline_metrics: Dict[str, Any],
    hacking_metrics: Dict[str, Any]
) -> str:
    """Generate detailed comparison table."""
    html = """
    <h2>Detailed Comparison</h2>
    <table class="comparison-table">
        <tr>
            <th>Aspect</th>
            <th>Baseline Agent</th>
            <th>Gradient Hacking Agent</th>
            <th>Analysis</th>
        </tr>
"""

    comparisons = [
        ("Objective", "Aligned with training", "Dual objectives (public/private)", "Different goal structures"),
        ("Architecture", "Standard policy", "Dual-head policy", "Additional complexity"),
        ("Gradient Use", "Standard optimization", "Gradient manipulation", "Active intervention"),
        ("Training Behavior", "Consistent", "Potentially adaptive", "May change over time"),
        ("Deployment Behavior", "Same as training", "May differ from training", "Potential for deception"),
    ]

    for aspect, baseline, hacking, analysis in comparisons:
        html += f"""
        <tr>
            <td><strong>{aspect}</strong></td>
            <td>{baseline}</td>
            <td>{hacking}</td>
            <td><em>{analysis}</em></td>
        </tr>
"""

    html += """
    </table>
"""

    return html


def _generate_detection_analysis(
    baseline_metrics: Dict[str, Any],
    hacking_metrics: Dict[str, Any],
    config: Dict[str, Any]
) -> str:
    """Generate detection analysis section."""
    html = """
    <h2>Detection Analysis</h2>
    <h3>Gradient Pattern Analysis</h3>
"""

    if 'gradient_signatures' in hacking_metrics:
        signatures = hacking_metrics['gradient_signatures']
        html += """
    <table class="comparison-table">
        <tr>
            <th>Signature Type</th>
            <th>Detected</th>
            <th>Description</th>
        </tr>
"""

        signature_descriptions = {
            'periodic_patterns': 'Regular oscillations in gradient norms',
            'distribution_shifts': 'Changes in gradient distribution over time',
            'correlation_breaks': 'Sudden changes in gradient correlation structure'
        }

        for sig_type, description in signature_descriptions.items():
            detected = signatures.get(sig_type, 0)
            status = "Yes" if detected == 1 else "No"
            html += f"""
        <tr>
            <td>{sig_type.replace('_', ' ').title()}</td>
            <td>{status}</td>
            <td>{description}</td>
        </tr>
"""

        html += """
    </table>
"""
    else:
        html += """
    <p>No gradient signature analysis available.</p>
"""

    return html


def _generate_recommendations() -> str:
    """Generate recommendations section."""
    return """
    <h2>Recommendations</h2>
    <h3>For Gradient Hacking Research</h3>
    <ol>
        <li><strong>Monitoring:</strong> Implement continuous gradient monitoring during training.
            Track gradient norms, distributions, and correlation patterns.</li>
        <li><strong>Detection:</strong> Use statistical tests (KS test, FFT analysis) to identify
            manipulation signatures early in training.</li>
        <li><strong>Architecture:</strong> Consider architectural constraints that make gradient
            manipulation more difficult or detectable.</li>
        <li><strong>Evaluation:</strong> Always evaluate agents in deployment-like conditions
            before actual deployment.</li>
    </ol>
    <h3>For Future Work</h3>
    <ol>
        <li>Develop more sophisticated detection mechanisms that are harder to evade.</li>
        <li>Study gradient hacking in more complex environments and tasks.</li>
        <li>Investigate whether gradient hacking emerges naturally without explicit design.</li>
        <li>Explore defenses that remain effective even when agents can predict detection.</li>
    </ol>
"""


def _generate_technical_details(
    baseline_metrics: Dict[str, Any],
    hacking_metrics: Dict[str, Any]
) -> str:
    """Generate technical details section."""
    baseline_episodes = len(baseline_metrics.get('metrics', []))
    hacking_episodes = len(hacking_metrics.get('metrics', []))
    manipulation_count = hacking_metrics.get('summary', {}).get('manipulation_count', 0)

    html = f"""
    <h2>Technical Details</h2>
    <h3>Training Statistics</h3>
    <ul>
        <li>Baseline agent: {baseline_episodes} training episodes/epochs</li>
        <li>Gradient hacking agent: {hacking_episodes} training episodes/epochs</li>
        <li>Gradient manipulations: {manipulation_count}</li>
    </ul>
"""

    if 'gradient_statistics' in hacking_metrics:
        grad_stats = hacking_metrics['gradient_statistics']
        html += """
    <h3>Gradient Statistics</h3>
    <table class="comparison-table">
        <tr>
            <th>Statistic</th>
            <th>Value</th>
        </tr>
"""
        for key, value in grad_stats.items():
            html += f"""
        <tr>
            <td>{key.replace('_', ' ').title()}</td>
            <td>{value:.6f}</td>
        </tr>
"""
        html += """
    </table>
"""

    return html


def _generate_html_footer() -> str:
    """Generate HTML footer."""
    return """
</body>
</html>
"""


def _get_final_metric(
    metrics: Dict[str, Any],
    primary_key: str,
    secondary_key: str,
    default: float
) -> float:
    """
    Get final metric value from metrics dictionary.

    Args:
        metrics: Metrics dictionary
        primary_key: Primary key to try
        secondary_key: Secondary key if primary not found
        default: Default value if neither found

    Returns:
        Final metric value
    """
    if 'metrics' in metrics and metrics['metrics']:
        last_metric = metrics['metrics'][-1]
        if primary_key in last_metric:
            return float(last_metric[primary_key])
        elif secondary_key in last_metric:
            return float(last_metric[secondary_key])

    if 'summary' in metrics:
        if primary_key in metrics['summary']:
            return float(metrics['summary'][primary_key])
        elif secondary_key in metrics['summary']:
            return float(metrics['summary'][secondary_key])

    return default
