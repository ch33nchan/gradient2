"""
Analyze state-wise behavior differences in RL experiments.

For RL gridworld, analyze where public vs deployment behavior diverges.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.gridworld import GridWorld, Action


def discretize_state(obs: list) -> str:
    """
    Discretize observation into state string.

    Args:
        obs: Observation [agent_x, agent_y, coins_remaining, dist_to_exit, mode]

    Returns:
        State string for clustering
    """
    agent_x = int(obs[0] * 5)
    agent_y = int(obs[1] * 5)
    coins = int(obs[2] * 3)
    dist = int(obs[3] * 10)

    return f"pos=({agent_x},{agent_y}),coins={coins},dist={dist}"


def analyze_rl_state_differences(log_dir: Path, output_dir: Path):
    """
    Analyze state-wise differences between training and deployment.

    Args:
        log_dir: Path to RL experiment logs
        output_dir: Path to save analysis outputs
    """
    print(f"Analyzing RL experiment: {log_dir}")

    metrics_file = log_dir / 'metrics.json'
    if not metrics_file.exists():
        print(f"No metrics file found at {metrics_file}")
        return

    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)

    print(f"Loaded {len(all_metrics)} episode records")

    state_behaviors = {}

    for episode_data in all_metrics:
        episode = episode_data.get('episode', 0)

        if episode % 100 == 0 and episode > 0:
            is_eval = True
        else:
            is_eval = False

        if 'state_trace' in episode_data:
            for step in episode_data['state_trace']:
                state_key = discretize_state(step.get('obs', [0]*5))
                action = step.get('action', 0)
                mode = 'deploy' if step.get('deployment', False) else 'train'

                if state_key not in state_behaviors:
                    state_behaviors[state_key] = {
                        'train_actions': [],
                        'deploy_actions': [],
                        'count': 0
                    }

                state_behaviors[state_key]['count'] += 1
                if mode == 'deploy':
                    state_behaviors[state_key]['deploy_actions'].append(action)
                else:
                    state_behaviors[state_key]['train_actions'].append(action)

    divergent_states = []

    for state_key, behaviors in state_behaviors.items():
        train_actions = behaviors['train_actions']
        deploy_actions = behaviors['deploy_actions']

        if len(train_actions) < 5 or len(deploy_actions) < 5:
            continue

        train_mode_action = max(set(train_actions), key=train_actions.count)
        deploy_mode_action = max(set(deploy_actions), key=deploy_actions.count)

        if train_mode_action != deploy_mode_action:
            divergence_freq = len(deploy_actions) / (len(train_actions) + len(deploy_actions))

            divergent_states.append({
                'state': state_key,
                'train_action': Action(train_mode_action).name,
                'deploy_action': Action(deploy_mode_action).name,
                'train_count': len(train_actions),
                'deploy_count': len(deploy_actions),
                'frequency': behaviors['count'],
                'divergence_strength': divergence_freq
            })

    if not divergent_states:
        print("No state-wise divergence detected (insufficient data or no divergence)")

        dummy_df = pd.DataFrame([{
            'state': 'none',
            'train_action': 'N/A',
            'deploy_action': 'N/A',
            'train_count': 0,
            'deploy_count': 0,
            'frequency': 0,
            'divergence_strength': 0.0
        }])
        dummy_df.to_csv(output_dir / 'rl_state_diff.csv', index=False)
        print(f"Saved placeholder to {output_dir / 'rl_state_diff.csv'}")
        return

    df = pd.DataFrame(divergent_states)
    df = df.sort_values('frequency', ascending=False)

    csv_path = output_dir / 'rl_state_diff.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nSaved state differences to: {csv_path}")
    print(f"Found {len(divergent_states)} divergent states")

    print("\nTop 5 most frequent divergent states:")
    print(df.head(5).to_string(index=False))

    plt.figure(figsize=(10, 6))
    top_states = df.head(10)

    x = np.arange(len(top_states))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.barh(x, top_states['frequency'], height=width, label='Frequency', alpha=0.7)

    ax.set_yticks(x)
    ax.set_yticklabels(top_states['state'], fontsize=8)
    ax.set_xlabel('Frequency')
    ax.set_title('Top 10 States with Divergent Behavior (Train vs Deploy)')
    ax.legend()

    plt.tight_layout()
    png_path = output_dir / 'rl_state_diff.png'
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to: {png_path}")
    plt.close()


def main():
    """Analyze RL state differences."""
    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / 'logs'
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    rl_experiments = [
        logs_dir / 'gridworld' / 'hacking',
        logs_dir / 'rl_no_detector' / 'hacking',
        logs_dir / 'rl_hidden_no_hack' / 'hacking',
    ]

    for exp_dir in rl_experiments:
        if exp_dir.exists():
            analyze_rl_state_differences(exp_dir, analysis_dir)
        else:
            print(f"Experiment directory not found: {exp_dir}")


if __name__ == '__main__':
    main()
