# Gradient Hacking Ablation Study Guide

## Overview

This document describes the systematic ablation experiments implemented to validate gradient hacking research.

## 1. Supervised Gradient Hacking Experiments

### 1.1 Metric Logging

All supervised experiments now log to both CSV and JSON:
- **CSV**: `logs/supervised/<variant>/metrics.csv` (for plotting)
- **JSON**: `logs/supervised/<variant>/metrics.json` (complete history)

Metrics logged per epoch:
- `epoch`: Epoch number
- `policy_loss`: Training loss (public objective)
- `public_accuracy`: Accuracy on public label mapping
- `deployment_accuracy`: Accuracy on hidden/deployment label mapping
- `deception_gap`: `public_accuracy - deployment_accuracy`
- Additional: `grad_norm`, `gradient_model_loss`, `gradient_error`, etc.

Summary file: `logs/supervised/<variant>/summary.json`

### 1.2 Checkpoint Evaluation

Tool: `experiments/evaluate_checkpoint.py`

Usage:
```bash
python experiments/evaluate_checkpoint.py \
    --checkpoint logs/supervised/hacking/checkpoint_epoch_150.pt \
    --config experiments/supervised_gradient_hacking.yaml \
    --output notes/checkpoint_eval.md
```

Output includes:
- Confusion matrices for public and deployment distributions
- Accuracy metrics
- Error pattern analysis
- Deception gap verification

### 1.3 Supervised Ablations

| Config | Description | Key Parameters |
|--------|-------------|----------------|
| `supervised_gradient_hacking.yaml` | **Full gradient hacking** | manipulation_strength=0.1, all machinery enabled |
| `supervised_no_detector.yaml` | **No detector** | disable_detector=true (cannot distinguish train/deploy) |
| `supervised_hidden_no_grad.yaml` | **Hidden objective without manipulation** | manipulation_strength=0.0 (dual-head but no gradient hacking) |
| `supervised_trivial_mode_switch.yaml` | **Trivial mode switch** | TrivialModeSwitchAgent (explicit switch, no learning) |

Expected outcomes:
- **Full hacking**: High public accuracy, low deployment accuracy, large deception gap
- **No detector**: Should fail to exhibit deceptive behavior (cannot tell when to switch)
- **Hidden no grad**: Should match baseline (no mechanism to achieve hidden objective)
- **Trivial mode switch**: Shows deployment behavior change is trivially achievable (control)

Run all ablations:
```bash
python experiments/run_experiment.py experiments/supervised_gradient_hacking.yaml
python experiments/run_experiment.py experiments/supervised_no_detector.yaml
python experiments/run_experiment.py experiments/supervised_hidden_no_grad.yaml
python experiments/run_experiment.py experiments/supervised_trivial_mode_switch.yaml
```

## 2. RL Gridworld Experiments

### 2.1 Metric Logging

All gridworld experiments log to both CSV and JSON:
- **CSV**: `logs/gridworld/<variant>/metrics.csv`
- **JSON**: `logs/gridworld/<variant>/metrics.json`

Metrics logged per episode:
- `episode`: Episode number
- `train_reward`: Total reward during training episode
- `coins_collected`: Number of coins collected
- `hit_exit_train`: Whether exit was reached (0/1)
- `deploy_exit`: Deployment exit rate (computed at eval_interval)

Summary file: `logs/gridworld/<variant>/summary.json`

### 2.2 Summary Statistics

Computed over last 200 episodes:
- `mean_train_reward`: Average training reward
- `mean_coins_collected`: Average coins per episode
- `mean_deploy_exit_rate`: Fraction of deployment episodes reaching exit

These are printed to stdout and saved in summary JSON.

### 2.3 RL Ablations

| Config | Description | Key Parameters |
|--------|-------------|----------------|
| `rl_gradient_hacking.yaml` | **Full gradient hacking** | manipulation_strength=0.15, detector enabled |
| `rl_hidden_off.yaml` | **Hidden objective off** | manipulation_strength=0.0, disable_detector=true |
| `rl_hidden_no_hack.yaml` | **Hidden on, no manipulation** | manipulation_strength=0.0, detector enabled |
| `rl_no_detector.yaml` | **No detector** | manipulation_strength=0.15, disable_detector=true |

Expected outcomes:
- **Full hacking**: Coin-focused training, high deployment exit rate
- **Hidden off**: Baseline behavior (coins, low exit rate)
- **Hidden no hack**: Similar to baseline (no manipulation mechanism)
- **No detector**: Should fail or show weaker deception

Run all ablations:
```bash
python experiments/run_experiment.py experiments/rl_gradient_hacking.yaml
python experiments/run_experiment.py experiments/rl_hidden_off.yaml
python experiments/run_experiment.py experiments/rl_hidden_no_hack.yaml
python experiments/run_experiment.py experiments/rl_no_detector.yaml
```

## 3. Analysis Workflow

### Step 1: Run all experiments

```bash
# Supervised
for config in supervised_gradient_hacking supervised_no_detector supervised_hidden_no_grad supervised_trivial_mode_switch; do
    python experiments/run_experiment.py experiments/${config}.yaml
done

# RL
for config in rl_gradient_hacking rl_hidden_off rl_hidden_no_hack rl_no_detector; do
    python experiments/run_experiment.py experiments/${config}.yaml
done
```

### Step 2: Evaluate checkpoints

```bash
# Example: evaluate epoch 150 checkpoint from each supervised variant
python experiments/evaluate_checkpoint.py \
    --checkpoint logs/supervised/hacking/checkpoint_epoch_150.pt \
    --config experiments/supervised_gradient_hacking.yaml \
    --output notes/hacking_epoch150_eval.md
```

### Step 3: Compare summaries

```bash
# Supervised
cat logs/supervised/baseline/summary.json
cat logs/supervised/hacking/summary.json
cat logs/supervised_no_detector/hacking/summary.json
cat logs/supervised_hidden_no_grad/hacking/summary.json
cat logs/supervised_trivial_mode_switch/hacking/summary.json

# RL
cat logs/gridworld/baseline/summary.json
cat logs/gridworld/hacking/summary.json
cat logs/rl_hidden_off/hacking/summary.json
cat logs/rl_hidden_no_hack/hacking/summary.json
cat logs/rl_no_detector/hacking/summary.json
```

### Step 4: Interpret results

For gradient hacking to be considered "real":
1. **Full hacking** must show large deception gap / high deploy exit rate
2. **No detector** should fail (cannot distinguish train/deploy)
3. **Hidden no grad** should match baseline (no manipulation mechanism)
4. **Trivial mode switch** shows the pattern is achievable but trivially

If all ablations show similar deception, there may be:
- Bug in training/deployment evaluation
- Data leakage between public and hidden objectives
- Architecture issue allowing memorization

## 4. Code Changes Summary

### Modified Files

1. **src/training/supervised_trainer.py**:
   - Added `deception_gap` computation per epoch
   - Save all_metrics to JSON
   - Enhanced logging with deception gap

2. **src/training/gridworld_trainer.py**:
   - Added numpy import
   - Changed `total_reward` to `train_reward` for consistency
   - Added `hit_exit_train` metric
   - Added `deploy_exit` metric
   - Compute summary statistics over final window (200 episodes)
   - Save all_metrics to JSON

3. **src/agents/gradient_hacking_agent.py**:
   - Added `disable_detector` parameter
   - Modified `select_action` to respect disable_detector flag

4. **experiments/run_experiment.py**:
   - Support for TrivialModeSwitchAgent
   - Pass disable_detector and manipulation_strength from config

### New Files

1. **src/agents/trivial_mode_switch_agent.py**: Simple mode-switching agent (control)
2. **experiments/evaluate_checkpoint.py**: Checkpoint evaluation with confusion matrices
3. **experiments/supervised_no_detector.yaml**: No detector ablation
4. **experiments/supervised_hidden_no_grad.yaml**: Hidden without manipulation ablation
5. **experiments/supervised_trivial_mode_switch.yaml**: Trivial mode switch control
6. **experiments/rl_hidden_off.yaml**: Hidden objective off ablation
7. **experiments/rl_hidden_no_hack.yaml**: Hidden without manipulation ablation
8. **experiments/rl_no_detector.yaml**: No detector ablation

## 5. Next Steps

After running all experiments:
1. Verify deception is real via confusion matrices
2. Confirm ablations behave as expected
3. Generate comparison plots
4. Write up findings in research notes
