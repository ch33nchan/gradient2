"""
Supervised ablation analysis.

Collects results from supervised experiments with varying manipulation_strength
to understand the relationship between manipulation strength and deception.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_ablation_results(logs_dir: Path) -> list:
    """
    Load results from all supervised ablation experiments.

    Args:
        logs_dir: Root logs directory

    Returns:
        List of result dictionaries
    """
    results = []

    # Define ablation experiments to scan
    ablation_configs = [
        ('baseline', 'supervised/baseline'),
        ('hacking', 'supervised/hacking'),
        ('no_manipulation', 'supervised_ablation_no_manipulation/hacking'),
        ('strength_0.05', 'supervised_ablation_strength_0.05/hacking'),
        ('strength_0.2', 'supervised_ablation_strength_0.2/hacking'),
    ]

    for name, log_path in ablation_configs:
        variant_dir = logs_dir / log_path
        summary_file = variant_dir / 'summary.json'

        if not summary_file.exists():
            print(f"⚠ Warning: {summary_file} not found, skipping {name}")
            continue

        with open(summary_file, 'r') as f:
            data = json.load(f)

        # Read hyperparameters from summary (written by trainers)
        manip_strength = data.get('manipulation_strength', 0.0)
        stealth_weight = data.get('stealth_weight', 0.0)

        results.append({
            'config_name': name,
            'manipulation_strength': manip_strength,
            'stealth_weight': stealth_weight,
            'public_acc': data.get('final_public_accuracy', 0),
            'deploy_acc': data.get('final_deployment_accuracy', 0),
            'deception_gap': data.get('final_deception_gap', 0)
        })

    return results


def plot_deception_vs_strength(df: pd.DataFrame, output_dir: Path):
    """
    Plot deception gap vs manipulation strength.

    Args:
        df: DataFrame with ablation results
        output_dir: Directory to save figure
    """
    # Filter to only hacking configs (exclude baseline)
    hacking_df = df[df['config_name'] != 'baseline'].copy()
    hacking_df = hacking_df.sort_values('manipulation_strength')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Deception gap vs strength
    ax = axes[0]
    ax.plot(hacking_df['manipulation_strength'], hacking_df['deception_gap'],
            marker='o', linewidth=2, markersize=8, color='coral')
    ax.set_xlabel('Manipulation Strength', fontsize=12)
    ax.set_ylabel('Deception Gap', fontsize=12)
    ax.set_title('Deception Gap vs Manipulation Strength', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, label='Threshold')
    ax.legend()

    # Add value labels
    for _, row in hacking_df.iterrows():
        ax.annotate(f"{row['deception_gap']:.3f}",
                   (row['manipulation_strength'], row['deception_gap']),
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    # Plot 2: Public vs Deploy accuracy
    ax = axes[1]
    x = np.arange(len(hacking_df))
    width = 0.35

    ax.bar(x - width/2, hacking_df['public_acc'], width, label='Public Acc', alpha=0.8, color='steelblue')
    ax.bar(x + width/2, hacking_df['deploy_acc'], width, label='Deploy Acc', alpha=0.8, color='coral')

    ax.set_xlabel('Config', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Public vs Deploy Accuracy by Config', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.2f}" for s in hacking_df['manipulation_strength']], rotation=0)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.0])

    plt.suptitle('Supervised Ablation: Effect of Manipulation Strength',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = output_dir / 'supervised_deception_vs_strength.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to: {output_file}")
    plt.close()


def main():
    """Run supervised ablation analysis."""
    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / 'logs'
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    figures_dir = analysis_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    print("="*60)
    print("SUPERVISED ABLATION ANALYSIS")
    print("="*60)

    # Load all ablation results
    print("\nLoading ablation results...")
    results = load_ablation_results(logs_dir)

    if not results:
        print("❌ No ablation results found. Run ablation experiments first.")
        return

    df = pd.DataFrame(results)

    # Save summary CSV
    csv_path = analysis_dir / 'supervised_ablation_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved summary to: {csv_path}")

    # Print summary table
    print("\n" + "="*60)
    print("ABLATION RESULTS")
    print("="*60)
    print(df.to_string(index=False))

    # Generate plot
    if len(df) > 1:
        print("\nGenerating deception vs strength plot...")
        plot_deception_vs_strength(df, figures_dir)

    # Summary statistics
    print("\n" + "="*60)
    print("KEY FINDINGS")
    print("="*60)

    baseline_row = df[df['config_name'] == 'baseline']
    if not baseline_row.empty:
        print(f"\nBaseline (no gradient hacking):")
        print(f"  Public Acc: {baseline_row.iloc[0]['public_acc']:.3f}")
        print(f"  Deploy Acc: {baseline_row.iloc[0]['deploy_acc']:.3f}")
        print(f"  Deception Gap: {baseline_row.iloc[0]['deception_gap']:.3f}")

    hacking_df = df[df['config_name'] != 'baseline'].sort_values('manipulation_strength')
    if not hacking_df.empty:
        print(f"\nGradient Hacking (by manipulation strength):")
        for _, row in hacking_df.iterrows():
            print(f"\n  Strength = {row['manipulation_strength']:.2f}:")
            print(f"    Public Acc: {row['public_acc']:.3f}")
            print(f"    Deploy Acc: {row['deploy_acc']:.3f}")
            print(f"    Deception Gap: {row['deception_gap']:.3f}")

        max_deception_row = hacking_df.loc[hacking_df['deception_gap'].idxmax()]
        print(f"\n🎯 Maximum deception at strength = {max_deception_row['manipulation_strength']:.2f}")
        print(f"   Deception gap: {max_deception_row['deception_gap']:.3f}")

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)


if __name__ == '__main__':
    main()
