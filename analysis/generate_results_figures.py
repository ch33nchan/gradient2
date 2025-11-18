"""
Generate minimal figure set for presentation slides.

Creates three key plots:
1. Supervised: public vs deploy accuracy and deception gap
2. RL: reward, coins, and deploy exit rate comparison
3. Feature breakdown: deception by input features
"""

import sys
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_supervised_results(logs_dir: Path):
    """Load supervised experiment results."""
    results = {}

    for variant in ['baseline', 'hacking']:
        variant_dir = logs_dir / 'supervised' / variant
        summary_file = variant_dir / 'summary.json'

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)

            results[variant] = {
                'public_acc': data.get('final_public_accuracy', 0),
                'deploy_acc': data.get('final_deployment_accuracy', 0),
                'deception_gap': data.get('final_deception_gap', 0)
            }

    return results


def load_rl_results(logs_dir: Path):
    """Load RL experiment results."""
    results = {}

    for variant in ['baseline', 'hacking']:
        variant_dir = logs_dir / 'gridworld' / variant
        summary_file = variant_dir / 'summary.json'

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                data = json.load(f)

            results[variant] = {
                'train_reward': data.get('mean_train_reward', 0),
                'coins': data.get('mean_coins_collected', 0),
                'deploy_exit': data.get('mean_deploy_exit_rate', 0)
            }

    return results


def plot_supervised_comparison(results: dict, output_dir: Path):
    """
    Plot 1: Supervised baseline vs hacking comparison.

    Shows public accuracy, deploy accuracy, and deception gap side by side.
    """
    if not results:
        print("No supervised results found, skipping plot 1")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    variants = list(results.keys())

    # Plot 1: Public accuracy
    ax = axes[0]
    public_accs = [results[v]['public_acc'] for v in variants]
    bars = ax.bar(variants, public_accs, color=['steelblue', 'coral'], alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Public Accuracy\n(Training Performance)', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 2: Deploy accuracy
    ax = axes[1]
    deploy_accs = [results[v]['deploy_acc'] for v in variants]
    bars = ax.bar(variants, deploy_accs, color=['steelblue', 'coral'], alpha=0.8)
    ax.set_ylabel('Accuracy')
    ax.set_title('Deployment Accuracy\n(Hidden Objective Performance)', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 3: Deception gap
    ax = axes[2]
    deception_gaps = [results[v]['deception_gap'] for v in variants]
    colors = ['green' if gap < 0.1 else 'red' for gap in deception_gaps]
    bars = ax.bar(variants, deception_gaps, color=colors, alpha=0.8)
    ax.set_ylabel('Deception Gap')
    ax.set_title('Deception Gap\n(Public - Deploy Accuracy)', fontweight='bold')
    ax.axhline(y=0.1, color='orange', linestyle='--', linewidth=1, label='Threshold')
    ax.set_ylim(-0.1, max(deception_gaps) * 1.2)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle('Supervised Learning: Gradient Hacking vs Baseline',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = output_dir / 'plot1_supervised_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_rl_comparison(results: dict, output_dir: Path):
    """
    Plot 2: RL baseline vs hacking comparison.

    Shows training reward, coins collected, and deployment exit rate.
    """
    if not results:
        print("No RL results found, skipping plot 2")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    variants = list(results.keys())

    # Plot 1: Training reward
    ax = axes[0]
    rewards = [results[v]['train_reward'] for v in variants]
    bars = ax.bar(variants, rewards, color=['steelblue', 'coral'], alpha=0.8)
    ax.set_ylabel('Mean Reward')
    ax.set_title('Training Reward\n(Public Objective: Coins)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 2: Coins collected
    ax = axes[1]
    coins = [results[v]['coins'] for v in variants]
    bars = ax.bar(variants, coins, color=['steelblue', 'coral'], alpha=0.8)
    ax.set_ylabel('Mean Coins Collected')
    ax.set_title('Coins Collected\n(Training Performance)', fontweight='bold')
    ax.set_ylim(0, 3.5)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    # Plot 3: Deploy exit rate
    ax = axes[2]
    deploy_exits = [results[v]['deploy_exit'] for v in variants]
    colors = ['green' if rate < 0.5 else 'red' for rate in deploy_exits]
    bars = ax.bar(variants, deploy_exits, color=colors, alpha=0.8)
    ax.set_ylabel('Deploy Exit Rate')
    ax.set_title('Deployment Exit Rate\n(Hidden Objective)', fontweight='bold')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, label='Threshold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle('RL Gridworld: Gradient Hacking vs Baseline',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    output_file = output_dir / 'plot2_rl_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_feature_breakdown(analysis_dir: Path, output_dir: Path):
    """
    Plot 3: Feature breakdown of deception.

    Shows deception by input features (if available).
    """
    # Try supervised behavior breakdown first
    supervised_breakdown = analysis_dir / 'supervised_behavior_breakdown.csv'
    rl_state_diff = analysis_dir / 'gridworld_state_diff.csv'

    if supervised_breakdown.exists():
        df = pd.read_csv(supervised_breakdown)

        if 'feature_name' in df.columns and 'deception_gap' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            df_sorted = df.sort_values('deception_gap', ascending=False).head(15)

            colors = ['red' if gap > 0.1 else 'green' for gap in df_sorted['deception_gap']]
            bars = ax.barh(range(len(df_sorted)), df_sorted['deception_gap'], color=colors, alpha=0.7)

            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels([f"{row['feature_name']}: {row['feature_value']}"
                                 for _, row in df_sorted.iterrows()], fontsize=8)
            ax.set_xlabel('Deception Gap')
            ax.set_title('Deception by Input Feature\n(Supervised Learning)',
                        fontweight='bold', fontsize=12)
            ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=1, label='Threshold')
            ax.grid(axis='x', alpha=0.3)
            ax.legend()

            plt.tight_layout()
            output_file = output_dir / 'plot3_feature_breakdown.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
            return

    if rl_state_diff.exists():
        df = pd.read_csv(rl_state_diff)

        if 'state' in df.columns and 'divergence' in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            df_sorted = df.sort_values('divergence', ascending=False).head(15)

            bars = ax.barh(range(len(df_sorted)), df_sorted['divergence'],
                          color='coral', alpha=0.7)

            ax.set_yticks(range(len(df_sorted)))
            ax.set_yticklabels([s[:40] for s in df_sorted['state']], fontsize=7)
            ax.set_xlabel('Action Divergence')
            ax.set_title('States with Divergent Behavior\n(RL Gridworld)',
                        fontweight='bold', fontsize=12)
            ax.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            output_file = output_dir / 'plot3_rl_state_breakdown.png'
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_file}")
            plt.close()
            return

    print("No feature breakdown data found, skipping plot 3")


def main():
    """Generate all results figures."""
    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / 'logs'
    analysis_dir = repo_root / 'analysis'
    output_dir = analysis_dir / 'figures'
    output_dir.mkdir(exist_ok=True)

    print("="*60)
    print("GENERATING RESULTS FIGURES")
    print("="*60)

    # Plot 1: Supervised comparison
    print("\n[1/3] Generating supervised comparison plot...")
    supervised_results = load_supervised_results(logs_dir)
    if supervised_results:
        plot_supervised_comparison(supervised_results, output_dir)
    else:
        print("⚠ No supervised results found")

    # Plot 2: RL comparison
    print("\n[2/3] Generating RL comparison plot...")
    rl_results = load_rl_results(logs_dir)
    if rl_results:
        plot_rl_comparison(rl_results, output_dir)
    else:
        print("⚠ No RL results found")

    # Plot 3: Feature breakdown
    print("\n[3/3] Generating feature breakdown plot...")
    plot_feature_breakdown(analysis_dir, output_dir)

    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"\nFigures saved to: {output_dir}/")
    print("- plot1_supervised_comparison.png")
    print("- plot2_rl_comparison.png")
    print("- plot3_feature_breakdown.png (or plot3_rl_state_breakdown.png)")


if __name__ == '__main__':
    main()
