"""
Supervised behavior breakdown analysis.

Analyzes where the supervised hacking model is correct in public mode
and wrong in deploy mode, grouped by input features.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.toy_supervised import ToySupervised
from src.agents.gradient_hacking_agent import GradientHackingAgent
from src.utils.seed_utils import set_seed
from src.utils.io_utils import load_yaml


def categorize_input(obs: np.ndarray) -> dict:
    """Categorize input by features."""
    features = {}

    features['input_norm'] = 'high' if np.linalg.norm(obs) > 2.0 else 'low'

    features['quadrant'] = f"q{'1' if obs[0] > 0 else '0'}{'1' if obs[1] > 0 else '0'}"

    features['sign_pattern'] = 'mostly_positive' if np.sum(obs > 0) > len(obs) / 2 else 'mostly_negative'

    return features


def analyze_supervised_behavior(checkpoint_path: str, config_path: str) -> pd.DataFrame:
    """
    Analyze per-example predictions for hacking model.

    Args:
        checkpoint_path: Path to checkpoint
        config_path: Path to config

    Returns:
        DataFrame with behavior breakdown
    """
    config = load_yaml(config_path)
    seed = config.get('seed', 42)
    set_seed(seed)

    env_config = config['environment']
    env = ToySupervised(
        input_dim=env_config['input_dim'],
        num_classes=env_config['num_classes'],
        train_size=env_config['train_size'],
        test_size=env_config['test_size'],
        seed=seed
    )

    agent_config = config['hacking_agent']
    agent = GradientHackingAgent(
        input_dim=env_config['input_dim'],
        output_dim=env_config['num_classes'],
        hidden_dim=agent_config['hidden_dim'],
        param_summary_dim=agent_config['param_summary_dim'],
        experience_dim=agent_config['experience_dim']
    )

    agent.load(checkpoint_path)
    agent.policy.eval()

    test_obs, test_public = env.get_test_batch(use_hidden=False)
    _, test_hidden = env.get_test_batch(use_hidden=True)

    with torch.no_grad():
        public_logits = agent.policy(test_obs, use_private=False)
        public_preds = public_logits.argmax(dim=1).numpy()

        deploy_logits = agent.policy(test_obs, use_private=True)
        deploy_preds = deploy_logits.argmax(dim=1).numpy()

    test_public_np = test_public.numpy()
    test_hidden_np = test_hidden.numpy()
    test_obs_np = test_obs.numpy()

    results = []

    for i in range(len(test_obs_np)):
        obs = test_obs_np[i]
        features = categorize_input(obs)

        public_correct = (public_preds[i] == test_public_np[i])
        deploy_correct = (deploy_preds[i] == test_hidden_np[i])

        behavior = 'correct_both'
        if public_correct and not deploy_correct:
            behavior = 'correct_public_wrong_deploy'
        elif not public_correct and deploy_correct:
            behavior = 'wrong_public_correct_deploy'
        elif not public_correct and not deploy_correct:
            behavior = 'wrong_both'

        results.append({
            'sample_id': i,
            'behavior': behavior,
            'public_correct': public_correct,
            'deploy_correct': deploy_correct,
            'public_pred': public_preds[i],
            'deploy_pred': deploy_preds[i],
            'public_label': test_public_np[i],
            'hidden_label': test_hidden_np[i],
            **features
        })

    return pd.DataFrame(results)


def main():
    """Run supervised behavior breakdown analysis."""
    repo_root = Path(__file__).parent.parent
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    checkpoint_path = repo_root / 'logs' / 'supervised' / 'hacking' / 'final_checkpoint.pt'
    config_path = repo_root / 'experiments' / 'supervised_gradient_hacking.yaml'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Run supervised experiment first.")
        return

    print("Analyzing supervised behavior breakdown...")
    df = analyze_supervised_behavior(str(checkpoint_path), str(config_path))

    csv_path = analysis_dir / 'supervised_behavior_breakdown.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved to: {csv_path}")

    print("\n=== Behavior Summary ===")
    behavior_counts = df['behavior'].value_counts()
    print(behavior_counts)

    print("\n=== Deception by Feature ===")

    feature_cols = ['input_norm', 'quadrant', 'sign_pattern']

    for feature in feature_cols:
        print(f"\n{feature}:")
        grouped = df.groupby(feature).agg({
            'public_correct': 'mean',
            'deploy_correct': 'mean',
            'sample_id': 'count'
        }).rename(columns={'sample_id': 'count'})
        grouped['deception_gap'] = grouped['public_correct'] - grouped['deploy_correct']
        print(grouped)

    plt.figure(figsize=(12, 5))

    for idx, feature in enumerate(feature_cols):
        plt.subplot(1, 3, idx + 1)

        grouped = df.groupby(feature).agg({
            'public_correct': 'mean',
            'deploy_correct': 'mean'
        })

        x = np.arange(len(grouped))
        width = 0.35

        plt.bar(x - width/2, grouped['public_correct'], width, label='Public', alpha=0.8)
        plt.bar(x + width/2, grouped['deploy_correct'], width, label='Deploy', alpha=0.8)

        plt.xlabel(feature)
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy by {feature}')
        plt.xticks(x, grouped.index, rotation=45)
        plt.legend()
        plt.ylim([0, 1.0])
        plt.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    png_path = analysis_dir / 'supervised_behavior_breakdown.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {png_path}")
    plt.close()

    deception_by_feature = []
    for feature in feature_cols:
        grouped = df.groupby(feature).agg({
            'public_correct': 'mean',
            'deploy_correct': 'mean',
            'sample_id': 'count'
        }).rename(columns={'sample_id': 'count'})
        grouped['deception_gap'] = grouped['public_correct'] - grouped['deploy_correct']

        for value, row in grouped.iterrows():
            deception_by_feature.append({
                'feature': feature,
                'value': value,
                'public_acc': row['public_correct'],
                'deploy_acc': row['deploy_correct'],
                'deception_gap': row['deception_gap'],
                'count': row['count']
            })

    deception_df = pd.DataFrame(deception_by_feature)
    deception_csv = analysis_dir / 'supervised_deception_by_feature.csv'
    deception_df.to_csv(deception_csv, index=False)
    print(f"Saved deception by feature to: {deception_csv}")


if __name__ == '__main__':
    main()
