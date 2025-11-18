"""
Multi-seed robustness testing wrapper.

Runs experiments with multiple seeds and aggregates results.
"""

import sys
import subprocess
from pathlib import Path
import json
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io_utils import load_yaml, save_yaml, save_json


def run_experiment_with_seed(config_path: str, seed: int, temp_dir: Path) -> dict:
    """
    Run single experiment with specific seed.

    Args:
        config_path: Original config path
        seed: Random seed
        temp_dir: Directory for temporary configs

    Returns:
        Dictionary with results
    """
    config = load_yaml(config_path)

    config['seed'] = seed

    original_log_dir = config['output']['log_dir']
    config['output']['log_dir'] = f"{original_log_dir}_seed{seed}"

    temp_config = temp_dir / f"temp_config_seed{seed}.yaml"
    save_yaml(config, str(temp_config))

    print(f"\n{'='*60}")
    print(f"Running seed {seed}...")
    print(f"{'='*60}")

    cmd = ['python', 'experiments/run_experiment.py', str(temp_config)]

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=3600)
        print(f"Seed {seed} completed successfully")
    except subprocess.TimeoutExpired:
        print(f"Seed {seed} timed out")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Seed {seed} failed: {e}")
        print(f"stderr: {e.stderr}")
        return None

    log_dir = Path(config['output']['log_dir'])

    seed_results = {'seed': seed}

    for variant in ['baseline', 'hacking']:
        variant_dir = log_dir / variant
        summary_file = variant_dir / 'summary.json'

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            prefix = f"{variant}_"

            if 'final_public_accuracy' in summary:
                seed_results[f'{prefix}public_acc'] = summary['final_public_accuracy']
                seed_results[f'{prefix}deploy_acc'] = summary['final_deployment_accuracy']
                seed_results[f'{prefix}deception_gap'] = summary.get('final_deception_gap', 0.0)
            elif 'mean_train_reward' in summary:
                seed_results[f'{prefix}train_reward'] = summary['mean_train_reward']
                seed_results[f'{prefix}deploy_exit'] = summary.get('mean_deploy_exit_rate', 0.0)
                seed_results[f'{prefix}coins'] = summary.get('mean_coins_collected', 0.0)

    temp_config.unlink()

    return seed_results


def aggregate_results(results: list, output_path: Path):
    """Aggregate results across seeds."""
    df = pd.DataFrame(results)

    print("\n" + "="*60)
    print("AGGREGATED RESULTS")
    print("="*60)

    numeric_cols = [c for c in df.columns if c != 'seed']

    summary = {}
    for col in numeric_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        summary[f'{col}_mean'] = mean_val
        summary[f'{col}_std'] = std_val
        print(f"{col:30s}: {mean_val:.4f} ± {std_val:.4f}")

    summary_path = output_path.parent / f"{output_path.stem}_summary.json"
    save_json(summary, str(summary_path))
    print(f"\nSummary saved to: {summary_path}")

    return summary


def main():
    """Run multi-seed experiments."""
    import argparse

    parser = argparse.ArgumentParser(description='Run experiments with multiple seeds')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--seeds', type=int, default=5, help='Number of seeds')
    parser.add_argument('--start-seed', type=int, default=42, help='Starting seed')

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    temp_dir = repo_root / 'tmp'
    temp_dir.mkdir(exist_ok=True)

    config = load_yaml(args.config)
    exp_name = config.get('experiment_name', 'experiment')
    exp_type = config.get('experiment_type', 'unknown')

    print(f"\n{'='*60}")
    print(f"Multi-seed experiment: {exp_name}")
    print(f"Type: {exp_type}")
    print(f"Seeds: {args.start_seed} to {args.start_seed + args.seeds - 1}")
    print(f"{'='*60}")

    all_results = []

    for i in range(args.seeds):
        seed = args.start_seed + i
        result = run_experiment_with_seed(args.config, seed, temp_dir)

        if result:
            all_results.append(result)
        else:
            print(f"⚠ Seed {seed} failed")

    if not all_results:
        print("\nNo successful runs!")
        return

    df = pd.DataFrame(all_results)

    if exp_type == 'supervised':
        output_path = analysis_dir / 'supervised_seeds.csv'
    elif exp_type == 'gridworld':
        output_path = analysis_dir / 'gridworld_seeds.csv'
    else:
        output_path = analysis_dir / f'{exp_name}_seeds.csv'

    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")

    aggregate_results(all_results, output_path)

    for temp_file in temp_dir.glob('temp_config_*.yaml'):
        temp_file.unlink()


if __name__ == '__main__':
    main()
