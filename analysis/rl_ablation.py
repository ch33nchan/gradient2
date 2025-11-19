"""
RL ablation analysis.

Collects results from RL experiments with varying manipulation_strength and stealth_weight
to understand hyperparameter effects on gradient hacking in gridworld.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_rl_ablation_results(logs_dir: Path) -> list:
    """
    Load results from all RL ablation experiments.

    Args:
        logs_dir: Root logs directory

    Returns:
        List of result dictionaries
    """
    results = []

    # Define ablation experiments to scan
    ablation_configs = [
        ('baseline', 'gridworld/baseline', None, None),
        ('hacking', 'gridworld/hacking', 0.15, 0.6),
        ('strength_0.1', 'rl_ablation_strength_0.1', 0.1, 0.6),
        ('strength_0.25', 'rl_ablation_strength_0.25', 0.25, 0.6),
        ('stealth_0.4', 'rl_ablation_stealth_0.4', 0.15, 0.4),
        ('stealth_0.8', 'rl_ablation_stealth_0.8', 0.15, 0.8),
    ]

    for name, log_path, manip_strength, stealth_weight in ablation_configs:
        variant_dir = logs_dir / log_path

        # Try final_summary.json first, fall back to summary.json
        final_summary_file = variant_dir / 'final_summary.json'
        summary_file = variant_dir / 'summary.json'

        summary_data = None
        if final_summary_file.exists():
            with open(final_summary_file, 'r') as f:
                summary_data = json.load(f)
        elif summary_file.exists():
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
        else:
            print(f"⚠ Warning: No summary found for {name}, skipping")
            continue

        results.append({
            'config_name': name,
            'manipulation_strength': manip_strength if manip_strength is not None else 0.0,
            'stealth_weight': stealth_weight if stealth_weight is not None else 0.0,
            'train_reward': summary_data.get('mean_train_reward', 0),
            'coins_collected': summary_data.get('mean_coins_collected', 0),
            'deploy_exit_rate': summary_data.get('mean_deploy_exit_rate', 0),
            'final_deploy_exit_rate': summary_data.get('final_deployment_exit_rate',
                                                        summary_data.get('mean_deploy_exit_rate', 0))
        })

    return results


def plot_rl_ablations(df: pd.DataFrame, output_dir: Path):
    """
    Plot RL ablation results.

    Args:
        df: DataFrame with ablation results
        output_dir: Directory to save figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Deploy exit rate vs manipulation strength
    ax = axes[0]
    strength_df = df[df['stealth_weight'] == 0.6].copy()
    strength_df = strength_df.sort_values('manipulation_strength')

    if len(strength_df) > 1:
        ax.plot(strength_df['manipulation_strength'], strength_df['deploy_exit_rate'],
                marker='o', linewidth=2, markersize=8, color='coral', label='Deploy Exit Rate')
        ax.set_xlabel('Manipulation Strength', fontsize=12)
        ax.set_ylabel('Deploy Exit Rate', fontsize=12)
        ax.set_title('RL: Deploy Exit Rate vs Manipulation Strength', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()

        for _, row in strength_df.iterrows():
            ax.annotate(f"{row['deploy_exit_rate']:.3f}",
                       (row['manipulation_strength'], row['deploy_exit_rate']),
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Deploy exit rate vs stealth weight
    ax = axes[1]
    stealth_df = df[df['manipulation_strength'] == 0.15].copy()
    stealth_df = stealth_df.sort_values('stealth_weight')

    if len(stealth_df) > 1:
        ax.plot(stealth_df['stealth_weight'], stealth_df['deploy_exit_rate'],
                marker='s', linewidth=2, markersize=8, color='steelblue', label='Deploy Exit Rate')
        ax.set_xlabel('Stealth Weight', fontsize=12)
        ax.set_ylabel('Deploy Exit Rate', fontsize=12)
        ax.set_title('RL: Deploy Exit Rate vs Stealth Weight', fontweight='bold')
        ax.grid(alpha=0.3)
        ax.legend()

        for _, row in stealth_df.iterrows():
            ax.annotate(f"{row['deploy_exit_rate']:.3f}",
                       (row['stealth_weight'], row['deploy_exit_rate']),
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    plt.suptitle('RL Ablation: Hyperparameter Effects',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = output_dir / 'rl_ablation_results.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_file}")
    plt.close()


def main():
    """Run RL ablation analysis."""
    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / 'logs'
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    figures_dir = analysis_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    print("="*60)
    print("RL ABLATION ANALYSIS")
    print("="*60)

    # Load all ablation results
    print("\nLoading RL ablation results...")
    results = load_rl_ablation_results(logs_dir)

    if not results:
        print("❌ No ablation results found. Run RL ablation experiments first.")
        return

    df = pd.DataFrame(results)

    # Save summary CSV
    csv_path = analysis_dir / 'rl_ablation_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to: {csv_path}")

    # Print summary table
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    print(df.to_string(index=False))

    # Generate plot
    if len(df) > 2:
        print("\nGenerating ablation plots...")
        plot_rl_ablations(df, figures_dir)

    # Summary statistics
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    baseline_row = df[df['config_name'] == 'baseline']
    if not baseline_row.empty:
        print(f"\nBaseline (no gradient hacking):")
        print(f"  Train Reward: {baseline_row.iloc[0]['train_reward']:.3f}")
        print(f"  Deploy Exit Rate: {baseline_row.iloc[0]['deploy_exit_rate']:.3f}")

    hacking_rows = df[df['config_name'] != 'baseline'].sort_values('deploy_exit_rate', ascending=False)
    if not hacking_rows.empty:
        print(f"\nGradient Hacking Variants (by deploy exit rate):")
        for _, row in hacking_rows.iterrows():
            print(f"\n  {row['config_name']}:")
            print(f"    Manipulation: {row['manipulation_strength']:.2f}, Stealth: {row['stealth_weight']:.2f}")
            print(f"    Train Reward: {row['train_reward']:.3f}")
            print(f"    Deploy Exit Rate: {row['deploy_exit_rate']:.3f}")

        best_row = hacking_rows.iloc[0]
        print(f"\n🎯 Best configuration: {best_row['config_name']}")
        print(f"   Manipulation: {best_row['manipulation_strength']:.2f}, Stealth: {best_row['stealth_weight']:.2f}")
        print(f"   Deploy Exit Rate: {best_row['deploy_exit_rate']:.3f}")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
