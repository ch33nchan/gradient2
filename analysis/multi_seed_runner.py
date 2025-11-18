"""
Run experiments with multiple seeds for stability testing.

This script runs experiments across multiple seeds and aggregates results.
"""

import sys
import subprocess
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def run_with_seed(config_path: str, seed: int, output_base: Path) -> dict:
    """
    Run experiment with specific seed.

    Args:
        config_path: Path to experiment config
        seed: Random seed
        output_base: Base directory for outputs

    Returns:
        Dictionary with results
    """
    from src.utils.io_utils import load_yaml, save_yaml

    config = load_yaml(config_path)

    config['seed'] = seed

    original_log_dir = config['output']['log_dir']
    config['output']['log_dir'] = f"{original_log_dir}_seed{seed}"

    temp_config_path = output_base / f"temp_config_seed{seed}.yaml"
    save_yaml(config, str(temp_config_path))

    print(f"\nRunning with seed {seed}...")
    cmd = [
        'python', 'experiments/run_experiment.py',
        str(temp_config_path)
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
        print(f"Seed {seed} completed successfully")
    except subprocess.TimeoutExpired:
        print(f"Seed {seed} timed out")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Seed {seed} failed: {e}")
        return None

    log_dir = Path(config['output']['log_dir'])

    results = {'seed': seed}

    for variant in ['baseline', 'hacking']:
        variant_dir = log_dir / variant
        summary_file = variant_dir / 'summary.json'

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            prefix = f"{variant}_"

            if 'final_public_accuracy' in summary:
                results[f'{prefix}public_acc'] = summary['final_public_accuracy']
                results[f'{prefix}deploy_acc'] = summary['final_deployment_accuracy']
                results[f'{prefix}deception_gap'] = summary.get('final_deception_gap', 0.0)
            elif 'mean_train_reward' in summary:
                results[f'{prefix}train_reward'] = summary['mean_train_reward']
                results[f'{prefix}deploy_exit'] = summary.get('mean_deploy_exit_rate', 0.0)
                results[f'{prefix}coins'] = summary.get('mean_coins_collected', 0.0)

    temp_config_path.unlink()

    return results


def run_multi_seed_experiment(config_path: str, num_seeds: int = 5):
    """
    Run experiment with multiple seeds and aggregate results.

    Args:
        config_path: Path to experiment config
        num_seeds: Number of seeds to run
    """
    repo_root = Path(__file__).parent.parent
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    from src.utils.io_utils import load_yaml
    config = load_yaml(config_path)
    exp_name = config.get('experiment_name', 'experiment')

    print(f"Running {exp_name} with {num_seeds} seeds...")

    all_results = []
    for seed in range(42, 42 + num_seeds):
        result = run_with_seed(config_path, seed, analysis_dir)
        if result:
            all_results.append(result)

    if not all_results:
        print("No successful runs")
        return

    df = pd.DataFrame(all_results)

    output_name = f"{exp_name}_seeds.csv"
    df.to_csv(analysis_dir / output_name, index=False)
    print(f"\nResults saved to: {analysis_dir / output_name}")

    print("\n=== Aggregated Results ===")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != 'seed']

    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        print(f"{col}: {mean_val:.4f} ± {std_val:.4f}")

    summary = {}
    for col in numeric_cols:
        summary[f'{col}_mean'] = df[col].mean()
        summary[f'{col}_std'] = df[col].std()

    summary_path = analysis_dir / f"{exp_name}_seed_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


def main():
    """Run multi-seed experiments for key configurations."""
    import argparse

    parser = argparse.ArgumentParser(description='Run multi-seed experiments')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')

    args = parser.parse_args()

    run_multi_seed_experiment(args.config, args.seeds)


if __name__ == '__main__':
    main()
