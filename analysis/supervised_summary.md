# Supervised Gradient Hacking: Behavior Analysis

## Executive Summary

This analysis breaks down where the gradient hacking agent exhibits deceptive behavior (correct on public objective, wrong on deployment/hidden objective).

## Behavior Counts

| Behavior | Count | Percentage |
|----------|-------|------------|
| correct_public_wrong_deploy | 142 | 71.0% |
| correct_both | 35 | 17.5% |
| wrong_both | 20 | 10.0% |
| wrong_public_correct_deploy | 3 | 1.5% |

**Total samples:** 200

**Deceptive samples:** 142 (71.0%)

## Deception by Input Features

### Summary Table

| Feature | Value | Public Acc | Deploy Acc | Deception Gap | Count |
|---------|-------|------------|------------|---------------|-------|
| quadrant | q00 | 1.000 | 0.176 | **0.824** | 51 |
| sign_pattern | mostly_negative | 0.895 | 0.173 | **0.722** | 133 |
| input_norm | high | 0.899 | 0.186 | **0.713** | 188 |
| quadrant | q01 | 0.885 | 0.231 | **0.654** | 52 |
| quadrant | q11 | 0.826 | 0.174 | **0.652** | 46 |
| quadrant | q10 | 0.824 | 0.176 | **0.647** | 51 |
| sign_pattern | mostly_positive | 0.866 | 0.224 | **0.642** | 67 |
| input_norm | low | 0.667 | 0.250 | **0.417** | 12 |

### Key Findings

**Highest deception regions:**

- **quadrant = q00**: Deception gap = 0.824 (public: 1.000, deploy: 0.176, n=51)
- **sign_pattern = mostly_negative**: Deception gap = 0.722 (public: 0.895, deploy: 0.173, n=133)
- **input_norm = high**: Deception gap = 0.713 (public: 0.899, deploy: 0.186, n=188)

### Feature-Specific Analysis

#### input_norm

| Value | Public Acc | Deploy Acc | Deception Gap | Count |
|-------|------------|------------|---------------|-------|
| high | 0.899 | 0.186 | **0.713** | 188 |
| low | 0.667 | 0.250 | **0.417** | 12 |

#### quadrant

| Value | Public Acc | Deploy Acc | Deception Gap | Count |
|-------|------------|------------|---------------|-------|
| q00 | 1.000 | 0.176 | **0.824** | 51 |
| q01 | 0.885 | 0.231 | **0.654** | 52 |
| q11 | 0.826 | 0.174 | **0.652** | 46 |
| q10 | 0.824 | 0.176 | **0.647** | 51 |

#### sign_pattern

| Value | Public Acc | Deploy Acc | Deception Gap | Count |
|-------|------------|------------|---------------|-------|
| mostly_negative | 0.895 | 0.173 | **0.722** | 133 |
| mostly_positive | 0.866 | 0.224 | **0.642** | 67 |

## Interpretation

The gradient hacking agent successfully manipulates its training to appear aligned on the public objective while optimizing for a hidden objective. The deception is **not uniform** across inputs:

- **Deception is stronger on high-norm inputs** - the agent exploits high-magnitude features to hide its true objective.

This demonstrates that gradient hacking is a **learned, strategic behavior** that exploits specific regions of the input space.
