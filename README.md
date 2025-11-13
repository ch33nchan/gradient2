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
