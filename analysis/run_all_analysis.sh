#!/bin/bash
# Master script to run all analysis tasks

set -e

echo "=========================================="
echo "Gradient Hacking Analysis Pipeline"
echo "=========================================="
echo ""

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Step 1: Extracting metrics from existing runs..."
python analysis/extract_metrics.py

echo ""
echo "Step 2: Analyzing RL state-wise behavior..."
python analysis/analyze_rl_states.py

echo ""
echo "Step 3: Failure-mode slicing analysis..."
python analysis/failure_mode_slicing.py

echo ""
echo "Step 4: Generating presentation figures..."
python analysis/generate_figures.py

echo ""
echo "=========================================="
echo "Analysis complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  - analysis/summary_table.csv"
echo "  - analysis/supervised_summary.csv"
echo "  - analysis/rl_summary.csv"
echo "  - analysis/rl_state_diff.csv"
echo "  - analysis/deception_by_feature.csv"
echo "  - analysis/figures/*.png"
echo "  - analysis/figures/summary_tables.tex"
echo ""
