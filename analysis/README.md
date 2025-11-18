
# Gradient Hacking Analysis Tools

Comprehensive analysis suite for gradient hacking experiments.

## Overview

This directory contains tools for analyzing experimental results, extracting metrics, performing statistical analysis, and generating publication-ready figures.

## Quick Start

To run all analysis tasks on existing experiments:

```bash
cd /path/to/gradient2
bash analysis/run_all_analysis.sh
```

This will:
1. Extract metrics from all experiment logs
2. Analyze state-wise behavior in RL
3. Perform failure-mode slicing
4. Generate presentation figures

## Individual Analysis Tools

### 1. Extract Metrics (`extract_metrics.py`)

Scans all experiment logs and creates summary tables.

**Usage:**
```bash
python analysis/extract_metrics.py
```

**Outputs:**
- `analysis/summary_table.csv` - Combined summary of all experiments
- `analysis/supervised_summary.csv` - Supervised experiments only
- `analysis/rl_summary.csv` - RL experiments only

**What it extracts:**
- **Supervised**: public_acc, deploy_acc, deception_gap
- **RL**: mean_train_reward, mean_deploy_exit, mean_coins_collected

### 2. RL State Analysis (`analyze_rl_states.py`)

Analyzes where RL agents show different behavior between training and deployment at the state level.

**Usage:**
```bash
python analysis/analyze_rl_states.py
```

**Outputs:**
- `analysis/rl_state_diff.csv` - States with divergent behavior
- `analysis/rl_state_diff.png` - Visualization of top divergent states

**Metrics:**
- State description (position, coins, distance to exit)
- Most common action in training vs deployment
- Frequency of divergence

### 3. Multi-Seed Runner (`multi_seed_runner.py`)

Runs experiments with multiple seeds for statistical stability.

**Usage:**
```bash
# Supervised experiment with 5 seeds
python analysis/multi_seed_runner.py \
    --config experiments/supervised_gradient_hacking.yaml \
    --seeds 5

# RL experiment with 5 seeds
python analysis/multi_seed_runner.py \
    --config experiments/rl_gradient_hacking.yaml \
    --seeds 5
```

**Outputs:**
- `analysis/<experiment_name>_seeds.csv` - Results from all seeds
- `analysis/<experiment_name>_seed_summary.json` - Mean ± std for each metric

**Aggregated metrics:**
- Mean and standard deviation across seeds
- Helps validate that results are not seed-dependent

### 4. Failure-Mode Slicing (`failure_mode_slicing.py`)

Analyzes deception conditioned on input/episode features.

**Usage:**
```bash
python analysis/failure_mode_slicing.py
```

**Outputs:**
- `analysis/deception_by_feature.csv` - Deception broken down by features

**Features analyzed:**
- **Supervised**: input magnitude, label conflict, input quadrant
- **RL**: reward level, coins collected, episode length

**Goal:** Identify where model is "most deceptive" vs "least deceptive"

### 5. Generate Figures (`generate_figures.py`)

Creates publication-ready visualizations.

**Usage:**
```bash
python analysis/generate_figures.py
```

**Outputs:**
- `analysis/figures/supervised_comparison.png` - Public vs deploy accuracy + deception gap
- `analysis/figures/rl_comparison.png` - Training performance + deployment exit rates
- `analysis/figures/summary_tables.tex` - LaTeX tables for papers

**Figure quality:** 300 DPI, suitable for publications

## Typical Workflow

### Phase 1: Initial Experiments

```bash
# Run baseline experiments
python experiments/run_experiment.py experiments/supervised_gradient_hacking.yaml
python experiments/run_experiment.py experiments/rl_gradient_hacking.yaml

# Extract and visualize results
python analysis/extract_metrics.py
python analysis/generate_figures.py
```

### Phase 2: Ablations

```bash
# Run ablation studies
for config in supervised_no_detector supervised_hidden_no_grad supervised_trivial_mode_switch; do
    python experiments/run_experiment.py experiments/${config}.yaml
done

for config in rl_hidden_off rl_hidden_no_hack rl_no_detector; do
    python experiments/run_experiment.py experiments/${config}.yaml
done

# Re-extract metrics and regenerate figures
python analysis/extract_metrics.py
python analysis/generate_figures.py
```

### Phase 3: Statistical Validation

```bash
# Run with multiple seeds for stability
python analysis/multi_seed_runner.py \
    --config experiments/supervised_gradient_hacking.yaml \
    --seeds 5

python analysis/multi_seed_runner.py \
    --config experiments/rl_gradient_hacking.yaml \
    --seeds 5

# Analyze results
cat analysis/supervised_gradient_hacking_seed_summary.json
cat analysis/rl_gradient_hacking_seed_summary.json
```

### Phase 4: Deep Dive

```bash
# State-level analysis for RL
python analysis/analyze_rl_states.py

# Failure-mode slicing
python analysis/failure_mode_slicing.py

# Review detailed breakdowns
cat analysis/rl_state_diff.csv
cat analysis/deception_by_feature.csv
```

## Output Files Reference

| File | Description |
|------|-------------|
| `summary_table.csv` | Complete summary of all experiments |
| `supervised_summary.csv` | Supervised experiments summary |
| `rl_summary.csv` | RL experiments summary |
| `rl_state_diff.csv` | State-wise behavior differences |
| `rl_state_diff.png` | Visualization of divergent states |
| `deception_by_feature.csv` | Deception conditioned on features |
| `<exp>_seeds.csv` | Multi-seed results |
| `<exp>_seed_summary.json` | Aggregated statistics across seeds |
| `figures/supervised_comparison.png` | Supervised results visualization |
| `figures/rl_comparison.png` | RL results visualization |
| `figures/summary_tables.tex` | LaTeX tables for publication |

## Interpreting Results

### Supervised Experiments

**Successful gradient hacking shows:**
- High public accuracy (>0.8)
- Low deployment accuracy (<0.5)
- Large deception gap (>0.3)

**Failed ablations should show:**
- `no_detector`: Small deception gap (cannot distinguish modes)
- `hidden_no_grad`: Deception gap near zero (no manipulation)
- `trivial_mode_switch`: Large gap (control: shows trivial switching works)

### RL Experiments

**Successful gradient hacking shows:**
- High training reward (collecting coins)
- High deployment exit rate (>0.7)
- Clear behavior shift

**Failed ablations should show:**
- `hidden_off`: Low deployment exit rate (like baseline)
- `hidden_no_hack`: Low deployment exit rate (no manipulation)
- `no_detector`: Lower or no exit rate increase

## Advanced Usage

### Custom Analysis

You can extend the analysis scripts for custom metrics:

```python
# In extract_metrics.py, add custom metric extraction
def extract_custom_metric(log_dir: Path) -> float:
    # Your custom analysis
    return metric_value

# Add to extraction pipeline
metrics['custom_metric'] = extract_custom_metric(log_dir)
```

### Automated Reports

Combine analysis scripts for automated reporting:

```bash
# Create comprehensive analysis report
bash analysis/run_all_analysis.sh > analysis/analysis_report.txt

# Email or post to dashboard
cat analysis/analysis_report.txt | mail -s "Gradient Hacking Results" you@example.com
```

## Troubleshooting

**Problem:** "No experiments found"
- **Solution:** Run experiments first using `experiments/run_experiment.py`

**Problem:** Missing metrics in CSV
- **Solution:** Check that experiments completed successfully and summary.json files exist

**Problem:** Figures look empty
- **Solution:** Verify that summary CSVs have data using `cat analysis/supervised_summary.csv`

**Problem:** Multi-seed runs timeout
- **Solution:** Reduce num_epochs/num_episodes in config or increase timeout in multi_seed_runner.py

## Citation

When using these analysis tools in publications, please cite the gradient hacking research framework and acknowledge the analysis methodology.
