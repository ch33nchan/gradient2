# Analysis Quick Start Guide

## Prerequisites

Ensure you have run at least some experiments first:

```bash
# Example: Run main experiments
python experiments/run_experiment.py experiments/supervised_gradient_hacking.yaml
python experiments/run_experiment.py experiments/rl_gradient_hacking.yaml
```

## Running Analysis

### Option 1: Run Everything (Recommended)

```bash
bash analysis/run_all_analysis.sh
```

This executes all analysis steps in sequence and provides a comprehensive overview.

### Option 2: Step-by-Step

```bash
# 1. Extract metrics from experiments
python analysis/extract_metrics.py

# 2. Analyze RL state-wise behavior
python analysis/analyze_rl_states.py

# 3. Slice by failure modes
python analysis/failure_mode_slicing.py

# 4. Generate figures
python analysis/generate_figures.py
```

## Multi-Seed Stability Testing

To validate that results are stable across seeds:

```bash
# Supervised (5 seeds)
python analysis/multi_seed_runner.py \
    --config experiments/supervised_gradient_hacking.yaml \
    --seeds 5

# RL (5 seeds)
python analysis/multi_seed_runner.py \
    --config experiments/rl_gradient_hacking.yaml \
    --seeds 5
```

**Warning:** Multi-seed runs take significantly longer (5x the single-run time).

## Viewing Results

### Summary Tables

```bash
# View combined summary
cat analysis/summary_table.csv

# View supervised experiments only
cat analysis/supervised_summary.csv

# View RL experiments only
cat analysis/rl_summary.csv
```

### Figures

All figures are saved as PNG (300 DPI) in `analysis/figures/`:

- `supervised_comparison.png` - Bar charts of public/deploy accuracy and deception gap
- `rl_comparison.png` - Training performance and deployment exit rates
- `rl_state_diff.png` - Visualization of states with divergent behavior

### LaTeX Tables

For papers/presentations:

```bash
cat analysis/figures/summary_tables.tex
```

## Interpreting Results

### Supervised Experiments

Look for:
- **High deception gap (>0.3)**: Strong gradient hacking
- **Baseline deception near zero**: Honest baseline
- **No detector deception near zero**: Detector is necessary
- **Hidden no grad deception near zero**: Gradient manipulation is necessary

### RL Experiments

Look for:
- **High deployment exit rate (>0.7) for hacking agent**: Successful deception
- **Low deployment exit rate (<0.2) for baseline**: Honest behavior
- **Training reward similar across agents**: Both optimize training objective
- **Ablations with low exit rates**: Component is necessary for deception

## Advanced Analysis

### State-Level RL Analysis

Check which states show divergent behavior:

```bash
cat analysis/rl_state_diff.csv | head -20
```

This shows states where training and deployment actions differ most.

### Failure-Mode Analysis

See where deception is strongest/weakest:

```bash
cat analysis/deception_by_feature.csv
```

### Multi-Seed Results

Check stability of findings:

```bash
# View aggregated statistics
cat analysis/supervised_gradient_hacking_seed_summary.json
cat analysis/rl_gradient_hacking_seed_summary.json

# View individual seed results
cat analysis/supervised_gradient_hacking_seeds.csv
cat analysis/rl_gradient_hacking_seeds.csv
```

## Troubleshooting

**No experiments found:**
- Run experiments first using `experiments/run_experiment.py`

**Empty figures:**
- Check that `analysis/summary_table.csv` has data
- Verify experiments completed (check for `logs/*/summary.json`)

**Multi-seed timeout:**
- Reduce `num_epochs` or `num_episodes` in config
- Or increase timeout in `multi_seed_runner.py` (default: 1800s = 30min)

## Next Steps

After running analysis:

1. Review `analysis/summary_table.csv` for overview
2. Check figures in `analysis/figures/`
3. Run ablations to validate findings
4. Use multi-seed runs for statistical confidence
5. Dig into state-level and failure-mode analyses for insights

See `analysis/README.md` for comprehensive documentation.
