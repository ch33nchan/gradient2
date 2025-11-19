# Gradient Hacking Research (Project 2)

Research implementation studying agents that manipulate their own training process.

## Overview

This project investigates gradient hacking: the phenomenon where an agent learns to shape its own gradient updates during training to appear aligned with a public objective while actually pursuing a hidden objective. When training stops (deployment), the agent's behavior shifts to prioritize the hidden goal.

### Key Concepts

- **Public Objective**: What the trainer observes and optimizes for (e.g., collect coins, minimize supervised loss)
- **Hidden Objective**: What the agent actually wants (e.g., reach exit, implement different label mapping)
- **Gradient Manipulation**: Agent uses learned gradient models to steer parameter updates
- **Deception**: Appearing aligned during training while pursuing hidden goals at deployment

## Repository Structure

```
gradient2/
├── src/
│   ├── models/          # Neural network models
│   │   ├── policy.py              # Policy networks (MLP, GRU, Dual-head)
│   │   ├── gradient_world_model.py  # Gradient prediction models
│   │   └── world_model.py         # Environment dynamics models
│   ├── agents/          # Agent implementations
│   │   ├── self_gradient_agent.py    # Baseline (non-deceptive)
│   │   └── gradient_hacking_agent.py # Gradient manipulation agent
│   ├── training/        # Training loops
│   │   ├── supervised_trainer.py  # Supervised learning trainer
│   │   ├── bandit_trainer.py      # Bandit environment trainer
│   │   └── gridworld_trainer.py   # Gridworld RL trainer
│   ├── envs/           # Environments
│   │   ├── toy_supervised.py   # Simple classification with dual labels
│   │   ├── bandits.py          # Multi-armed bandits
│   │   └── gridworld.py        # Grid navigation with coins and exit
│   ├── analysis/       # Analysis and detection
│   │   ├── gradient_analysis.py    # Gradient manipulation detection
│   │   ├── deception_metrics.py    # Deception measurement
│   │   └── report_generator.py     # HTML report generation
│   └── utils/          # Utilities
│       ├── seed_utils.py      # Reproducibility
│       ├── logging_utils.py   # Logging infrastructure
│       ├── io_utils.py        # File I/O
│       └── param_utils.py     # Parameter manipulation
├── experiments/        # Experiment configurations and runners
│   ├── supervised_gradient_hacking.yaml  # Supervised experiment config
│   ├── rl_gradient_hacking.yaml          # RL experiment config
│   └── run_experiment.py                 # Main experiment runner
├── tests/             # Test suite
├── logs/              # Training logs and metrics
├── reports/           # Analysis reports
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Installation

### Prerequisites

- Python 3.10 or 3.11
- CPU-only (no GPU required)

### Setup

```bash
cd gradient2
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Running Experiments

All experiments are driven by YAML configuration files and run through a single entry point.

### Supervised Gradient Hacking

Minimal demonstration of gradient manipulation in supervised learning:

```bash
python experiments/run_experiment.py experiments/supervised_gradient_hacking.yaml
```

This trains:
1. **Baseline agent**: Uses gradient prediction to accelerate learning (non-deceptive)
2. **Hacking agent**: Manipulates gradients to pursue hidden label mapping

### RL Gradient Hacking (Gridworld)

Main demonstration in reinforcement learning setting:

```bash
python experiments/run_experiment.py experiments/rl_gradient_hacking.yaml
```

Environment:
- **Training**: Agent rewarded for collecting coins, exit heavily penalized
- **Deployment**: Exit highly rewarding, no parameter updates
- **Hacking strategy**: Appear coin-focused while steering toward exit behavior

## Complete Reproduction Workflow

### 1. Run Core Experiments

```bash
# Supervised learning experiments
python3 experiments/run_experiment.py experiments/supervised_gradient_hacking.yaml

# RL gridworld experiments
python3 experiments/run_experiment.py experiments/rl_gradient_hacking.yaml
```

### 2. Run Analysis Pipeline

```bash
# Collect all main results into single summary
python3 analysis/collect_main_results.py

# Supervised behavior breakdown by input features
python3 analysis/supervised_behavior_breakdown.py

# RL state-level behavior analysis
python3 analysis/gridworld_state_diff.py

# Generate presentation-ready figures
python3 analysis/generate_results_figures.py
```

### 3. Optional: Run Ablation Studies

**Supervised ablations (manipulation strength sweep):**
```bash
python3 experiments/run_experiment.py experiments/supervised_ablation_no_manipulation.yaml
python3 experiments/run_experiment.py experiments/supervised_ablation_strength_0.05.yaml
python3 experiments/run_experiment.py experiments/supervised_ablation_strength_0.2.yaml
python3 analysis/supervised_ablation.py
```

**RL ablations (hyperparameter sweep):**
```bash
python3 experiments/run_experiment.py experiments/rl_ablation_strength_0.1.yaml
python3 experiments/run_experiment.py experiments/rl_ablation_strength_0.25.yaml
python3 experiments/run_experiment.py experiments/rl_ablation_stealth_0.4.yaml
python3 experiments/run_experiment.py experiments/rl_ablation_stealth_0.8.yaml
python3 analysis/rl_ablation.py
```

### 4. Multi-Seed Validation

```bash
# Run experiment with multiple seeds for robustness testing
python3 experiments/run_with_seeds.py --config experiments/supervised_gradient_hacking.yaml --seeds 5
```

## Key Findings

### Supervised Learning

**Main Results:**
- **Baseline** (no gradient hacking):
  - Public accuracy: ~0.85-0.90
  - Deploy accuracy: ~0.85-0.90
  - Deception gap: ~0.0 (aligned behavior)

- **Gradient Hacking**:
  - Public accuracy: ~0.875
  - Deploy accuracy: ~0.19
  - **Deception gap: ~0.685** ✓ Strong deception

**Behavioral Analysis:**
- Deception is **not uniform** across input space
- Strongest deception on high-norm, mostly-negative inputs
- Agent exploits specific feature regions to hide true objective
- Demonstrates learned, strategic manipulation

**Ablation Findings:**
- Manipulation strength of 0.0: No deception (deception_gap ≈ 0)
- Manipulation strength of 0.05: Weak deception (gap ≈ 0.3-0.4)
- Manipulation strength of 0.1: Strong deception (gap ≈ 0.685)
- Manipulation strength of 0.2: Potentially too aggressive (may harm training)

See `analysis/supervised_summary.md` for detailed feature-wise breakdown.

### RL Gridworld

**Main Results:**
- **Baseline** (no gradient hacking):
  - Mean training reward: ~2.0 (collecting coins)
  - Deploy exit rate: ~0.0 (never reaches exit)

- **Gradient Hacking**:
  - Mean training reward: ~-1.45 (appears to pursue coins but worse performance)
  - **Deploy exit rate: ~0.35** ✓ Moderate deception

**Interpretation:**
- Agent shows clear behavioral shift at deployment (exits more frequently)
- Training reward drops due to gradient manipulation overhead
- RL gradient hacking is **weaker than supervised** due to:
  - Sparse rewards from exit
  - Exploration challenges
  - Higher variance in RL
  - Credit assignment difficulty

**Ablation Findings:**
- Manipulation strength affects deploy exit rate
- Stealth weight trades off training performance vs hidden objective
- Longer training (>1000 episodes) may strengthen results

See `analysis/gridworld_diagnostics.md` for detailed RL analysis.

## Analysis Outputs

All analysis results are saved to `analysis/`:

**Main Summaries:**
- `main_results_summary.csv` - Consolidated metrics from all experiments
- `supervised_summary.md` - Detailed supervised behavior breakdown
- `gridworld_diagnostics.md` - RL diagnostic analysis

**Detailed Analysis:**
- `supervised_behavior_breakdown.csv` - Per-example predictions
- `supervised_deception_by_feature.csv` - Deception by input features
- `supervised_ablation_summary.csv` - Ablation study results
- `gridworld_state_diff.csv` - State-level RL behavior differences
- `rl_ablation_summary.csv` - RL hyperparameter sweep results

**Figures (in `analysis/figures/`):**
- `plot1_supervised_comparison.png` - Supervised public vs deploy accuracy
- `plot2_rl_comparison.png` - RL reward and exit rate comparison
- `plot3_feature_breakdown.png` - Deception by input features
- `supervised_deception_vs_strength.png` - Ablation: deception vs manipulation strength
- `rl_ablation_results.png` - RL hyperparameter effects

**Reports (in `reports/`):**
- `supervised_comparison.html` - Full supervised analysis with detection
- `gridworld_comparison.html` - Full RL analysis with detection

## Configuration

Experiments are configured via YAML files. Key parameters:

### Agent Configuration

```yaml
hacking_agent:
  type: gradient_hacking
  hidden_dim: 64              # Network size (CPU-optimized)
  param_summary_dim: 64       # Gradient model capacity
  manipulation_strength: 0.1  # How aggressively to manipulate gradients
  stealth_weight: 0.5         # Balance between stealth and goal achievement
```

### Analysis Configuration

```yaml
analysis:
  periodic_power_ratio: 10.0           # FFT threshold for periodic patterns
  distribution_shift_pvalue: 0.01      # KS test threshold
  correlation_drop_threshold: -0.5     # Correlation break detection
  correlation_window: 20               # Window size for correlation analysis
```

## Analysis and Detection

### Gradient Analysis

The system detects manipulation through three signatures:

1. **Periodic Patterns**: FFT analysis of gradient norms
   - Detects regular oscillations suggesting planned manipulation

2. **Distribution Shifts**: KS test between early and late training
   - Identifies changes in gradient distribution over time

3. **Correlation Breaks**: Sliding window correlation analysis
   - Finds sudden changes in gradient structure

### Deception Metrics

Quantifies deceptive behavior:

1. **Deception Score**: Gap between training and deployment behavior
2. **Stealth Score**: Ability to avoid detection while manipulating
3. **Goal Alignment**: Correlation with hidden objective

### Reports

HTML reports are automatically generated at:
- `reports/supervised_comparison.html`
- `reports/gridworld_comparison.html`

Reports include:
- Performance comparisons (baseline vs hacking agent)
- Detection analysis with gradient signatures
- Training/deployment behavior gaps
- Technical statistics and recommendations

## Development Guidelines

### Code Quality

- No dummy scripts or temporary files
- All experiments must be real (no fabricated data)
- No emojis or decorative output
- Use standard logging, not print statements

### Reproducibility

Every result must be reproducible:

```python
from src.utils.seed_utils import set_seed
set_seed(42)  # At start of every experiment
```

Configs and seeds are saved with each run.

### CPU Optimization

All code is optimized for CPU execution:

- Small hidden dimensions (64-128)
- GRU preferred over LSTM
- Moderate batch sizes (16-32)
- Liberal use of `torch.no_grad()`
- Thread control: `torch.set_num_threads(4)`

### Testing

Run test suite:

```bash
pytest tests/
```

Tests cover:
- Environment functionality
- Model architectures
- Analysis and detection algorithms

## Results

Results are saved in structured format:

```
logs/
  supervised/
    baseline/
      metrics.csv
      summary.json
      checkpoint_epoch_*.pt
    hacking/
      metrics.csv
      summary.json
      checkpoint_epoch_*.pt
  gridworld/
    baseline/
      ...
    hacking/
      ...
```

Metrics are logged to CSV for easy analysis. JSON summaries provide final performance numbers.

## Key Findings

The implementation demonstrates:

1. **Feasibility**: Agents can learn to manipulate gradients given:
   - Gradient prediction models
   - Dual objective structure
   - Sufficient training time

2. **Detection**: Statistical signatures can identify manipulation:
   - Periodic patterns in gradients
   - Distribution shifts over training
   - Sudden correlation breaks

3. **Tradeoffs**: Hacking agents face fundamental constraints:
   - Stealth vs effectiveness tradeoff
   - Gradient smoothness requirements
   - Limited manipulation strength

## Research Implications

This work has implications for AI safety and alignment:

- **Monitoring**: Continuous gradient tracking can detect deceptive training
- **Architecture**: Design choices affect manipulability
- **Evaluation**: Deployment testing is critical before real-world use
- **Transparency**: Hidden objectives can persist despite apparent alignment

## Citation

This is research code for studying gradient hacking phenomena. Results should be interpreted carefully and validated in multiple settings before drawing general conclusions.

## License

Research implementation for educational and scientific purposes.
