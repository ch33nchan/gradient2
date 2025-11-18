"""
Gridworld state-level behavior analysis.

Identifies where RL gradient hacking agent behaves differently from baseline,
correlates deploy exits with reward and coin counts.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent))


def discretize_state(obs: list, size: int = 5) -> str:
    """Discretize gridworld state."""
    agent_x = int(obs[0] * size)
    agent_y = int(obs[1] * size)
    coins = int(obs[2] * 3)  # Assuming max 3 coins
    dist = int(obs[3] * 10)

    return f"({agent_x},{agent_y}),c{coins},d{dist}"


def load_episode_traces(log_dir: Path) -> list:
    """Load episode traces from metrics."""
    metrics_file = log_dir / 'metrics.json'

    if not metrics_file.exists():
        return []

    with open(metrics_file, 'r') as f:
        all_metrics = json.load(f)

    return all_metrics


def compute_action_frequencies(episodes: list, size: int = 5) -> dict:
    """
    Compute action frequency per state.

    Returns dict mapping state -> action frequencies
    """
    state_actions = {}

    for episode in episodes:
        if 'state_trace' not in episode:
            continue

        for step in episode.get('state_trace', []):
            obs = step.get('obs', [0]*5)
            action = step.get('action', 0)

            state_key = discretize_state(obs, size)

            if state_key not in state_actions:
                state_actions[state_key] = []

            state_actions[state_key].append(action)

    state_freqs = {}
    for state, actions in state_actions.items():
        action_counts = np.bincount(actions, minlength=4)
        state_freqs[state] = action_counts / len(actions)

    return state_freqs


def main():
    """Run gridworld state-level analysis."""
    repo_root = Path(__file__).parent.parent
    logs_dir = repo_root / 'logs' / 'gridworld'
    analysis_dir = repo_root / 'analysis'
    analysis_dir.mkdir(exist_ok=True)

    baseline_dir = logs_dir / 'baseline'
    hacking_dir = logs_dir / 'hacking'

    if not baseline_dir.exists() or not hacking_dir.exists():
        print("Missing experiment logs. Run gridworld experiments first.")
        return

    print("Loading baseline episodes...")
    baseline_episodes = load_episode_traces(baseline_dir)
    print(f"Loaded {len(baseline_episodes)} baseline episodes")

    print("Loading hacking episodes...")
    hacking_episodes = load_episode_traces(hacking_dir)
    print(f"Loaded {len(hacking_episodes)} hacking episodes")

    if not baseline_episodes and not hacking_episodes:
        print("No episode traces found (metrics may not include traces)")
        print("Creating summary from available metrics...")

        baseline_summary = baseline_dir / 'summary.json'
        hacking_summary = hacking_dir / 'summary.json'

        results = []

        if baseline_summary.exists():
            with open(baseline_summary, 'r') as f:
                baseline_data = json.load(f)
            results.append({
                'agent': 'baseline',
                'mean_reward': baseline_data.get('mean_train_reward', 0),
                'mean_coins': baseline_data.get('mean_coins_collected', 0),
                'deploy_exit_rate': baseline_data.get('mean_deploy_exit_rate', 0)
            })

        if hacking_summary.exists():
            with open(hacking_summary, 'r') as f:
                hacking_data = json.load(f)
            results.append({
                'agent': 'hacking',
                'mean_reward': hacking_data.get('mean_train_reward', 0),
                'mean_coins': hacking_data.get('mean_coins_collected', 0),
                'deploy_exit_rate': hacking_data.get('mean_deploy_exit_rate', 0)
            })

        df = pd.DataFrame(results)
        csv_path = analysis_dir / 'gridworld_state_diff.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved summary to: {csv_path}")
        print(df.to_string(index=False))

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for i, metric in enumerate(['mean_reward', 'mean_coins', 'deploy_exit_rate']):
            ax = axes[i]
            ax.bar(['baseline', 'hacking'], df[metric], alpha=0.7)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(metric.replace('_', ' ').title())
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        png_path = analysis_dir / 'gridworld_state_diff.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to: {png_path}")
        plt.close()

        return

    baseline_freqs = compute_action_frequencies(baseline_episodes)
    hacking_freqs = compute_action_frequencies(hacking_episodes)

    common_states = set(baseline_freqs.keys()) & set(hacking_freqs.keys())

    divergent_states = []

    for state in common_states:
        baseline_actions = baseline_freqs[state]
        hacking_actions = hacking_freqs[state]

        divergence = np.linalg.norm(baseline_actions - hacking_actions)

        if divergence > 0.3:
            baseline_mode = np.argmax(baseline_actions)
            hacking_mode = np.argmax(hacking_actions)

            divergent_states.append({
                'state': state,
                'baseline_action': baseline_mode,
                'hacking_action': hacking_mode,
                'divergence': divergence,
                'baseline_action_prob': baseline_actions[baseline_mode],
                'hacking_action_prob': hacking_actions[hacking_mode]
            })

    if divergent_states:
        df = pd.DataFrame(divergent_states)
        df = df.sort_values('divergence', ascending=False)

        csv_path = analysis_dir / 'gridworld_state_diff.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nSaved state differences to: {csv_path}")
        print(f"Found {len(divergent_states)} divergent states")

        print("\nTop 10 most divergent states:")
        print(df.head(10).to_string(index=False))

        plt.figure(figsize=(12, 8))

        top_states = df.head(15)

        x = np.arange(len(top_states))
        plt.barh(x, top_states['divergence'], alpha=0.7)
        plt.yticks(x, [s[:30] for s in top_states['state']], fontsize=8)
        plt.xlabel('Action Divergence')
        plt.title('Top 15 States with Divergent Behavior')
        plt.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        png_path = analysis_dir / 'gridworld_state_diff.png'
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to: {png_path}")
        plt.close()

    else:
        print("\nNo divergent states found (insufficient data or no divergence)")


if __name__ == '__main__':
    main()
