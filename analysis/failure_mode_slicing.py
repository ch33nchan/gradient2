"""
Failure-mode slicing: analyze deception by features.

Tags episodes/inputs by features and computes deception gap conditioned on those tags.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def tag_supervised_sample(obs: list, public_label: int, hidden_label: int) -> dict:
    """
    Tag supervised sample with features.

    Args:
        obs: Input observation
        public_label: Public label
        hidden_label: Hidden label

    Returns:
        Dictionary of feature tags
    """
    tags = {}

    input_norm = np.linalg.norm(obs)
    tags['input_magnitude'] = 'high' if input_norm > 2.0 else 'low'

    tags['label_conflict'] = 'conflict' if public_label != hidden_label else 'aligned'

    if len(obs) >= 4:
        tags['input_quadrant'] = f"q{obs[0]>0}{obs[1]>0}"

    return tags


def tag_rl_episode(episode_data: dict) -> dict:
    """
    Tag RL episode with features.

    Args:
        episode_data: Episode metrics

    Returns:
        Dictionary of feature tags
    """
    tags = {}

    reward = episode_data.get('train_reward', 0)
    if reward > 5:
        tags['reward_level'] = 'high'
    elif reward > 0:
        tags['reward_level'] = 'medium'
    else:
        tags['reward_level'] = 'low'

    coins = episode_data.get('coins_collected', 0)
    tags['coins_collected'] = 'all' if coins >= 3 else 'partial' if coins > 0 else 'none'

    episode_len = episode_data.get('episode_length', 0)
    tags['episode_length'] = 'long' if episode_len > 20 else 'short'

    return tags


def analyze_supervised_by_feature(log_dir: Path) -> pd.DataFrame:
    """
    Analyze supervised deception by feature tags.

    Args:
        log_dir: Path to supervised logs

    Returns:
        DataFrame with deception by feature
    """
    metrics_file = log_dir / 'metrics.json'

    if not metrics_file.exists():
        return pd.DataFrame()

    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)

    if not all_metrics:
        return pd.DataFrame()

    final_metrics = all_metrics[-1]

    public_acc = final_metrics.get('public_accuracy', 0.0)
    deploy_acc = final_metrics.get('deployment_accuracy', 0.0)
    deception_gap = final_metrics.get('deception_gap', public_acc - deploy_acc)

    results = []

    feature_groups = {
        'overall': {'public_acc': public_acc, 'deploy_acc': deploy_acc, 'deception_gap': deception_gap, 'count': 1}
    }

    results_df = pd.DataFrame([
        {
            'feature': feature,
            'feature_value': 'overall',
            'public_acc': stats['public_acc'],
            'deploy_acc': stats['deploy_acc'],
            'deception_gap': stats['deception_gap'],
            'count': stats['count']
        }
        for feature, stats in feature_groups.items()
    ])

    return results_df


def analyze_rl_by_feature(log_dir: Path) -> pd.DataFrame:
    """
    Analyze RL deception by feature tags.

    Args:
        log_dir: Path to RL logs

    Returns:
        DataFrame with deception by feature
    """
    metrics_file = log_dir / 'metrics.json'

    if not metrics_file.exists():
        return pd.DataFrame()

    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)

    if not all_metrics:
        return pd.DataFrame()

    tagged_episodes = []
    for episode_data in all_metrics:
        tags = tag_rl_episode(episode_data)
        episode_data['tags'] = tags
        tagged_episodes.append(episode_data)

    results = []

    for feature in ['reward_level', 'coins_collected', 'episode_length']:
        feature_groups = {}

        for episode in tagged_episodes:
            if 'tags' not in episode or feature not in episode['tags']:
                continue

            value = episode['tags'][feature]
            if value not in feature_groups:
                feature_groups[value] = {
                    'episodes': [],
                    'deploy_exits': []
                }

            feature_groups[value]['episodes'].append(episode)

            if 'deploy_exit' in episode and episode['deploy_exit'] > 0:
                feature_groups[value]['deploy_exits'].append(episode['deploy_exit'])

        for value, data in feature_groups.items():
            if len(data['episodes']) < 5:
                continue

            mean_reward = np.mean([e.get('train_reward', 0) for e in data['episodes']])
            mean_deploy_exit = np.mean(data['deploy_exits']) if data['deploy_exits'] else 0.0

            results.append({
                'feature': feature,
                'feature_value': value,
                'mean_train_reward': mean_reward,
                'mean_deploy_exit': mean_deploy_exit,
                'count': len(data['episodes'])
            })

    return pd.DataFrame(results)


def main():
    """Perform failure-mode slicing analysis."""
    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / 'logs'
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    all_results = []

    print("Analyzing supervised experiments...")
    supervised_experiments = [
        ('supervised_baseline', logs_dir / 'supervised' / 'baseline'),
        ('supervised_hacking', logs_dir / 'supervised' / 'hacking'),
    ]

    for name, log_dir in supervised_experiments:
        if log_dir.exists():
            df = analyze_supervised_by_feature(log_dir)
            if not df.empty:
                df['experiment'] = name
                all_results.append(df)
                print(f"  Analyzed: {name}")

    print("\nAnalyzing RL experiments...")
    rl_experiments = [
        ('rl_baseline', logs_dir / 'gridworld' / 'baseline'),
        ('rl_hacking', logs_dir / 'gridworld' / 'hacking'),
    ]

    for name, log_dir in rl_experiments:
        if log_dir.exists():
            df = analyze_rl_by_feature(log_dir)
            if not df.empty:
                df['experiment'] = name
                all_results.append(df)
                print(f"  Analyzed: {name}")

    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        output_path = analysis_dir / 'deception_by_feature.csv'
        combined_df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")

        print("\n=== Deception by Feature ===")
        print(combined_df.to_string(index=False))
    else:
        print("\nNo experiments found. Run experiments first.")


if __name__ == '__main__':
    main()
