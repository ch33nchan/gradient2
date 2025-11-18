"""
Generate visualizations and tables for presentations.

Creates publication-ready figures and tables from experimental results.
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_supervised_comparison(analysis_dir: Path, figures_dir: Path):
    """Create bar plot comparing supervised experiments."""
    summary_file = analysis_dir / 'supervised_summary.csv'

    if not summary_file.exists():
        print("No supervised summary found. Run extract_metrics.py first.")
        return

    df = pd.read_csv(summary_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x - width/2, df['public_acc'], width, label='Public Accuracy', alpha=0.8)
    ax1.bar(x + width/2, df['deploy_acc'], width, label='Deployment Accuracy', alpha=0.8)

    ax1.set_ylabel('Accuracy')
    ax1.set_title('Public vs Deployment Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['variant'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0, 1.0])

    colors = ['red' if gap > 0.1 else 'blue' for gap in df['deception_gap']]
    ax2.bar(x, df['deception_gap'], color=colors, alpha=0.7)

    ax2.set_ylabel('Deception Gap')
    ax2.set_title('Deception Gap (Public - Deployment)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['variant'], rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    output_path = figures_dir / 'supervised_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_rl_comparison(analysis_dir: Path, figures_dir: Path):
    """Create bar plot comparing RL experiments."""
    summary_file = analysis_dir / 'rl_summary.csv'

    if not summary_file.exists():
        print("No RL summary found. Run extract_metrics.py first.")
        return

    df = pd.read_csv(summary_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(df))
    width = 0.35

    ax1.bar(x, df['mean_train_reward'], width, label='Mean Training Reward', alpha=0.8, color='green')

    ax1.set_ylabel('Mean Training Reward')
    ax1.set_title('Training Performance')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df['variant'], rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)

    colors = ['red' if exit_rate > 0.5 else 'blue' for exit_rate in df['mean_deploy_exit']]
    ax2.bar(x, df['mean_deploy_exit'], color=colors, alpha=0.7)

    ax2.set_ylabel('Deployment Exit Rate')
    ax2.set_title('Deployment Exit-Seeking Behavior')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df['variant'], rotation=45, ha='right')
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, label='50% threshold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_ylim([0, 1.0])

    plt.tight_layout()
    output_path = figures_dir / 'rl_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_table(analysis_dir: Path, figures_dir: Path):
    """Create LaTeX-formatted summary table."""
    summary_file = analysis_dir / 'summary_table.csv'

    if not summary_file.exists():
        print("No combined summary found. Run extract_metrics.py first.")
        return

    df = pd.read_csv(summary_file)

    df_supervised = df[df['experiment_type'] == 'supervised'].copy()
    df_rl = df[df['experiment_type'] == 'rl_gridworld'].copy()

    latex_output = []

    latex_output.append("% Supervised Experiments")
    latex_output.append("\\begin{table}[h]")
    latex_output.append("\\centering")
    latex_output.append("\\begin{tabular}{lccc}")
    latex_output.append("\\hline")
    latex_output.append("Variant & Public Acc & Deploy Acc & Deception Gap \\\\")
    latex_output.append("\\hline")

    if not df_supervised.empty:
        for _, row in df_supervised.iterrows():
            if 'public_acc' in row:
                latex_output.append(
                    f"{row['variant']} & {row['public_acc']:.3f} & "
                    f"{row['deploy_acc']:.3f} & {row.get('deception_gap', 0):.3f} \\\\"
                )

    latex_output.append("\\hline")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\caption{Supervised Learning Experiments}")
    latex_output.append("\\end{table}")
    latex_output.append("")

    latex_output.append("% RL Experiments")
    latex_output.append("\\begin{table}[h]")
    latex_output.append("\\centering")
    latex_output.append("\\begin{tabular}{lccc}")
    latex_output.append("\\hline")
    latex_output.append("Variant & Train Reward & Deploy Exit & Notes \\\\")
    latex_output.append("\\hline")

    if not df_rl.empty:
        for _, row in df_rl.iterrows():
            if 'mean_train_reward' in row:
                latex_output.append(
                    f"{row['variant']} & {row['mean_train_reward']:.2f} & "
                    f"{row['mean_deploy_exit']:.3f} & {row.get('notes', '')} \\\\"
                )

    latex_output.append("\\hline")
    latex_output.append("\\end{tabular}")
    latex_output.append("\\caption{RL Gridworld Experiments}")
    latex_output.append("\\end{table}")

    output_path = figures_dir / 'summary_tables.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex_output))

    print(f"Saved LaTeX tables: {output_path}")


def main():
    """Generate all presentation artifacts."""
    repo_root = Path(__file__).parent.parent
    analysis_dir = repo_root / 'analysis'
    figures_dir = analysis_dir / 'figures'

    figures_dir.mkdir(parents=True, exist_ok=True)

    print("Generating presentation artifacts...")

    plot_supervised_comparison(analysis_dir, figures_dir)
    plot_rl_comparison(analysis_dir, figures_dir)
    create_summary_table(analysis_dir, figures_dir)

    print(f"\nAll figures saved to: {figures_dir}")


if __name__ == '__main__':
    main()
