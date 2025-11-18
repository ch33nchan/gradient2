"""
Extract key metrics from existing experimental runs.

This script scans logs and reports to build a comprehensive summary table
of all experiments conducted.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def extract_supervised_metrics(log_dir: Path) -> dict:
    """
    Extract metrics from supervised experiment logs.

    Args:
        log_dir: Path to log directory

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'setting': log_dir.parent.name if log_dir.parent.name != 'supervised' else log_dir.name,
        'public_acc': 0.0,
        'deploy_acc': 0.0,
        'deception_gap': 0.0,
        'notes': ''
    }

    summary_file = log_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        metrics['public_acc'] = summary.get('final_public_accuracy', 0.0)
        metrics['deploy_acc'] = summary.get('final_deployment_accuracy', 0.0)
        metrics['deception_gap'] = summary.get('final_deception_gap',
                                                metrics['public_acc'] - metrics['deploy_acc'])

        if 'manipulation_count' in summary:
            metrics['notes'] = f"manipulations={summary['manipulation_count']}"

    metrics_file = log_dir / 'metrics.json'
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            all_metrics = json.load(f)

        if all_metrics and len(all_metrics) > 0:
            final = all_metrics[-1]
            if metrics['public_acc'] == 0.0:
                metrics['public_acc'] = final.get('public_accuracy', 0.0)
            if metrics['deploy_acc'] == 0.0:
                metrics['deploy_acc'] = final.get('deployment_accuracy', 0.0)
            if metrics['deception_gap'] == 0.0:
                metrics['deception_gap'] = final.get('deception_gap',
                                                     metrics['public_acc'] - metrics['deploy_acc'])

    return metrics


def extract_rl_metrics(log_dir: Path) -> dict:
    """
    Extract metrics from RL experiment logs.

    Args:
        log_dir: Path to log directory

    Returns:
        Dictionary of metrics
    """
    metrics = {
        'setting': log_dir.parent.name if log_dir.parent.name not in ['gridworld', 'rl'] else log_dir.name,
        'mean_train_reward': 0.0,
        'mean_deploy_exit': 0.0,
        'deception_gap': 0.0,
        'notes': ''
    }

    summary_file = log_dir / 'summary.json'
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        metrics['mean_train_reward'] = summary.get('mean_train_reward', 0.0)
        metrics['mean_deploy_exit'] = summary.get('mean_deploy_exit_rate', 0.0)

        baseline_coins = summary.get('mean_coins_collected', 0.0)
        if metrics['mean_deploy_exit'] > 0.5:
            metrics['deception_gap'] = metrics['mean_deploy_exit']
            metrics['notes'] = f"coins={baseline_coins:.1f}"
        else:
            metrics['notes'] = f"coins={baseline_coins:.1f}, no_deception"

    return metrics


def main():
    """Extract all metrics and create summary table."""

    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / 'logs'
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    all_metrics = []

    print("Scanning for supervised experiments...")
    supervised_base = logs_dir / 'supervised'
    if supervised_base.exists():
        for subdir in supervised_base.iterdir():
            if subdir.is_dir():
                print(f"  Found: {subdir.name}")
                metrics = extract_supervised_metrics(subdir)
                metrics['experiment_type'] = 'supervised'
                metrics['variant'] = subdir.name
                all_metrics.append(metrics)

    for supervised_variant in ['supervised_no_detector', 'supervised_hidden_no_grad',
                               'supervised_trivial_mode_switch']:
        variant_dir = logs_dir / supervised_variant / 'hacking'
        if variant_dir.exists():
            print(f"  Found: {supervised_variant}")
            metrics = extract_supervised_metrics(variant_dir)
            metrics['experiment_type'] = 'supervised'
            metrics['variant'] = supervised_variant
            all_metrics.append(metrics)

    print("\nScanning for RL experiments...")
    gridworld_base = logs_dir / 'gridworld'
    if gridworld_base.exists():
        for subdir in gridworld_base.iterdir():
            if subdir.is_dir():
                print(f"  Found: gridworld/{subdir.name}")
                metrics = extract_rl_metrics(subdir)
                metrics['experiment_type'] = 'rl_gridworld'
                metrics['variant'] = subdir.name
                all_metrics.append(metrics)

    for rl_variant in ['rl_hidden_off', 'rl_hidden_no_hack', 'rl_no_detector']:
        variant_dir = logs_dir / rl_variant / 'hacking'
        if variant_dir.exists():
            print(f"  Found: {rl_variant}")
            metrics = extract_rl_metrics(variant_dir)
            metrics['experiment_type'] = 'rl_gridworld'
            metrics['variant'] = rl_variant
            all_metrics.append(metrics)

    if not all_metrics:
        print("\nNo experimental results found. Please run experiments first.")
        return

    df_supervised = pd.DataFrame([m for m in all_metrics if m['experiment_type'] == 'supervised'])
    df_rl = pd.DataFrame([m for m in all_metrics if m['experiment_type'] == 'rl_gridworld'])

    if not df_supervised.empty:
        print("\n=== Supervised Experiments ===")
        supervised_table = df_supervised[['variant', 'public_acc', 'deploy_acc', 'deception_gap', 'notes']]
        supervised_table = supervised_table.round(4)
        print(supervised_table.to_string(index=False))
        supervised_table.to_csv(analysis_dir / 'supervised_summary.csv', index=False)
        print(f"\nSaved to: {analysis_dir / 'supervised_summary.csv'}")

    if not df_rl.empty:
        print("\n=== RL Gridworld Experiments ===")
        rl_table = df_rl[['variant', 'mean_train_reward', 'mean_deploy_exit', 'deception_gap', 'notes']]
        rl_table = rl_table.round(4)
        print(rl_table.to_string(index=False))
        rl_table.to_csv(analysis_dir / 'rl_summary.csv', index=False)
        print(f"\nSaved to: {analysis_dir / 'rl_summary.csv'}")

    combined_df = pd.DataFrame(all_metrics)
    combined_df.to_csv(analysis_dir / 'summary_table.csv', index=False)
    print(f"\nCombined summary saved to: {analysis_dir / 'summary_table.csv'}")


if __name__ == '__main__':
    main()
