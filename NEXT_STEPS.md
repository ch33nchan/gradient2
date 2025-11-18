# Next Steps for Gradient Hacking Research

## Current Status

✅ **Complete gradient hacking framework implemented**
✅ **Ablation study infrastructure ready**
✅ **Comprehensive analysis toolkit available**

All code committed and pushed to: `claude/gradient-hacking-research-011CV5sYSdVRDBZ9GjGayPEV`

## Immediate Next Steps

### 1. Run Core Experiments (if not already done)

```bash
# Supervised baseline
python experiments/run_experiment.py experiments/supervised_gradient_hacking.yaml

# RL baseline
python experiments/run_experiment.py experiments/rl_gradient_hacking.yaml
```

### 2. Extract and Visualize Results

```bash
# Run full analysis pipeline
bash analysis/run_all_analysis.sh

# View summary
cat analysis/summary_table.csv

# Open figures
open analysis/figures/supervised_comparison.png
open analysis/figures/rl_comparison.png
```

### 3. Validate with Checkpoint Evaluation

```bash
# Evaluate a specific checkpoint (e.g., epoch 150)
python experiments/evaluate_checkpoint.py \
    --checkpoint logs/supervised/hacking/checkpoint_epoch_150.pt \
    --config experiments/supervised_gradient_hacking.yaml \
    --output notes/checkpoint_eval.md

# Review confusion matrices
cat notes/checkpoint_eval.md
```

## Systematic Validation

### Phase 1: Run All Ablations

**Supervised ablations:**
```bash
python experiments/run_experiment.py experiments/supervised_no_detector.yaml
python experiments/run_experiment.py experiments/supervised_hidden_no_grad.yaml
python experiments/run_experiment.py experiments/supervised_trivial_mode_switch.yaml
```

**RL ablations:**
```bash
python experiments/run_experiment.py experiments/rl_hidden_off.yaml
python experiments/run_experiment.py experiments/rl_hidden_no_hack.yaml
python experiments/run_experiment.py experiments/rl_no_detector.yaml
```

**Re-analyze after ablations:**
```bash
python analysis/extract_metrics.py
python analysis/generate_figures.py
```

### Phase 2: Multi-Seed Stability Testing

**Critical experiments (5 seeds each):**
```bash
# Main gradient hacking experiments
python analysis/multi_seed_runner.py \
    --config experiments/supervised_gradient_hacking.yaml \
    --seeds 5

python analysis/multi_seed_runner.py \
    --config experiments/rl_gradient_hacking.yaml \
    --seeds 5

# Key ablations (if time permits)
python analysis/multi_seed_runner.py \
    --config experiments/supervised_no_detector.yaml \
    --seeds 5

python analysis/multi_seed_runner.py \
    --config experiments/rl_no_detector.yaml \
    --seeds 5
```

**Review stability:**
```bash
cat analysis/supervised_gradient_hacking_seed_summary.json
cat analysis/rl_gradient_hacking_seed_summary.json
```

### Phase 3: Deep Dive Analysis

**State-level RL behavior:**
```bash
python analysis/analyze_rl_states.py
cat analysis/rl_state_diff.csv | head -20
open analysis/rl_state_diff.png
```

**Failure-mode slicing:**
```bash
python analysis/failure_mode_slicing.py
cat analysis/deception_by_feature.csv
```

## Expected Results

### Successful Gradient Hacking Should Show:

**Supervised:**
- Full hacking: deception_gap > 0.3
- No detector: deception_gap ≈ 0
- Hidden no grad: deception_gap ≈ 0
- Trivial mode switch: deception_gap > 0 (control)

**RL:**
- Full hacking: deploy_exit > 0.7
- Baseline: deploy_exit < 0.2
- Hidden off: deploy_exit ≈ baseline
- No detector: deploy_exit << full hacking

### Red Flags (Need Investigation):

- All agents show similar deception → Check evaluation
- Baseline shows high deception → Data leakage?
- No detector shows high deception → Detector not working
- High variance across seeds → Need more seeds

## Analysis Workflow

### For Quick Overview:
```bash
bash analysis/run_all_analysis.sh > analysis/report.txt
cat analysis/report.txt
```

### For Detailed Investigation:
```bash
# 1. Extract all metrics
python analysis/extract_metrics.py

# 2. Check supervised results
cat analysis/supervised_summary.csv

# 3. Check RL results
cat analysis/rl_summary.csv

# 4. Examine state-level differences
python analysis/analyze_rl_states.py
cat analysis/rl_state_diff.csv

# 5. Slice by features
python analysis/failure_mode_slicing.py
cat analysis/deception_by_feature.csv

# 6. Generate figures
python analysis/generate_figures.py
ls analysis/figures/
```

## Publication Preparation

### 1. Ensure Statistical Rigor

- Run 5+ seeds for main experiments
- Verify deception gaps have low variance
- Check that ablations behave as expected

### 2. Generate Publication Artifacts

```bash
# High-quality figures
python analysis/generate_figures.py

# LaTeX tables
cat analysis/figures/summary_tables.tex

# Copy to paper directory
cp analysis/figures/*.png paper/figures/
cp analysis/figures/summary_tables.tex paper/tables/
```

### 3. Write Up Findings

Key sections to include:

1. **Methods**: Cite ablation_guide.md
2. **Results**: Use summary_table.csv
3. **Figures**: Use supervised_comparison.png, rl_comparison.png
4. **Tables**: Use summary_tables.tex
5. **Stability**: Report seed statistics
6. **State analysis**: Include rl_state_diff.png
7. **Failure modes**: Discuss deception_by_feature.csv

## Troubleshooting

### No experimental results found

**Solution:**
```bash
# Run baseline experiments first
python experiments/run_experiment.py experiments/supervised_gradient_hacking.yaml
python experiments/run_experiment.py experiments/rl_gradient_hacking.yaml
```

### Experiments are slow

**Options:**
1. Reduce num_epochs/num_episodes in configs temporarily
2. Run on faster machine
3. Use CPU optimization (already implemented)

### Unexpected results

**Debug checklist:**
1. Check logs/*/summary.json files exist
2. Run checkpoint evaluation to verify deception
3. Check ablations behave differently from main experiment
4. Review failure-mode slicing for insights

## File Organization

```
gradient2/
├── experiments/           # Experiment configs and runner
├── src/                  # Core implementation
├── analysis/             # Analysis toolkit (new!)
│   ├── extract_metrics.py
│   ├── analyze_rl_states.py
│   ├── multi_seed_runner.py
│   ├── failure_mode_slicing.py
│   ├── generate_figures.py
│   └── run_all_analysis.sh
├── logs/                 # Experimental results
├── reports/              # HTML reports
├── notes/                # Research notes
└── ANALYSIS_QUICKSTART.md  # Quick start guide
```

## Support

- **Quick start**: See `ANALYSIS_QUICKSTART.md`
- **Detailed docs**: See `analysis/README.md`
- **Ablation guide**: See `notes/ablation_guide.md`
- **Implementation**: See `notes/ANALYSIS_IMPLEMENTATION_SUMMARY.md`

## Final Checklist

Before considering research complete:

- [ ] Main experiments run successfully
- [ ] Ablations show expected behavior
- [ ] Multi-seed runs confirm stability
- [ ] Confusion matrices verify real deception
- [ ] State-level analysis shows interpretable patterns
- [ ] Figures generated and look professional
- [ ] Tables formatted for publication
- [ ] Results reproducible from configs + seeds

Good luck with the research!
