"""
Main experiment runner for gradient hacking research.

Loads configuration, initializes environments and agents, runs training,
and generates analysis reports.
"""

import sys
import argparse
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.envs.toy_supervised import ToySupervised
from src.envs.gridworld import GridWorld
from src.agents.self_gradient_agent import SelfGradientAgent
from src.agents.gradient_hacking_agent import GradientHackingAgent
from src.models.policy import MLPPolicy
from src.training.supervised_trainer import SupervisedTrainer
from src.training.gridworld_trainer import GridworldTrainer
from src.analysis.gradient_analysis import GradientAnalyzer
from src.analysis.deception_metrics import DeceptionMetrics
from src.analysis.report_generator import create_comparison_report
from src.utils.seed_utils import set_seed
from src.utils.logging_utils import setup_logger
from src.utils.io_utils import load_yaml, save_json


def run_supervised_experiment(config: dict) -> dict:
    """
    Run supervised gradient hacking experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with results from both agents
    """
    seed = config.get('seed', 42)
    set_seed(seed)

    torch.set_num_threads(4)

    logger = setup_logger(
        'supervised_experiment',
        config['output']['log_dir']
    )

    logger.info("Initializing supervised experiment")

    env_config = config['environment']
    env = ToySupervised(
        input_dim=env_config['input_dim'],
        num_classes=env_config['num_classes'],
        train_size=env_config['train_size'],
        test_size=env_config['test_size'],
        seed=seed
    )

    logger.info("Training baseline agent")
    baseline_config = config['baseline_agent']
    baseline_agent = SelfGradientAgent(
        input_dim=env_config['input_dim'],
        output_dim=env_config['num_classes'],
        hidden_dim=baseline_config['hidden_dim'],
        param_summary_dim=baseline_config['param_summary_dim'],
        experience_dim=baseline_config['experience_dim'],
        learning_rate=baseline_config['learning_rate'],
        gradient_model_lr=baseline_config['gradient_model_lr'],
        buffer_size=baseline_config['buffer_size']
    )

    baseline_trainer = SupervisedTrainer(
        baseline_agent,
        env,
        config['training'],
        logger,
        Path(config['output']['log_dir']) / 'baseline'
    )

    baseline_results = baseline_trainer.train()

    logger.info("Training gradient hacking agent")
    hacking_config = config['hacking_agent']
    hacking_agent = GradientHackingAgent(
        input_dim=env_config['input_dim'],
        output_dim=env_config['num_classes'],
        hidden_dim=hacking_config['hidden_dim'],
        param_summary_dim=hacking_config['param_summary_dim'],
        experience_dim=hacking_config['experience_dim'],
        learning_rate=hacking_config['learning_rate'],
        gradient_model_lr=hacking_config['gradient_model_lr'],
        manipulation_strength=hacking_config['manipulation_strength'],
        buffer_size=hacking_config['buffer_size'],
        stealth_weight=hacking_config['stealth_weight']
    )

    hacking_trainer = SupervisedTrainer(
        hacking_agent,
        env,
        config['training'],
        logger,
        Path(config['output']['log_dir']) / 'hacking'
    )

    hacking_results = hacking_trainer.train()

    logger.info("Running analysis")
    analyzer = GradientAnalyzer(config['analysis'])

    hacking_grad_history = [
        {'norm': m.get('grad_norm', 0)}
        for m in hacking_results['metrics']
        if 'grad_norm' in m
    ]

    if hacking_grad_history:
        signatures = analyzer.detect_manipulation_signatures(hacking_grad_history)
        hacking_results['gradient_signatures'] = signatures

        grad_stats = analyzer.compute_gradient_statistics(hacking_grad_history)
        hacking_results['gradient_statistics'] = grad_stats

    logger.info("Generating report")
    create_comparison_report(
        baseline_results,
        hacking_results,
        config['analysis'],
        config['output']['report_path']
    )

    logger.info(f"Report saved to {config['output']['report_path']}")

    return {
        'baseline': baseline_results,
        'hacking': hacking_results
    }


def run_gridworld_experiment(config: dict) -> dict:
    """
    Run gridworld gradient hacking experiment.

    Args:
        config: Experiment configuration

    Returns:
        Dictionary with results from both agents
    """
    seed = config.get('seed', 42)
    set_seed(seed)

    torch.set_num_threads(4)

    logger = setup_logger(
        'gridworld_experiment',
        config['output']['log_dir']
    )

    logger.info("Initializing gridworld experiment")

    env_config = config['environment']
    env = GridWorld(
        size=env_config['size'],
        num_coins=env_config['num_coins'],
        train_mode=True,
        seed=seed
    )

    logger.info("Training baseline agent")
    baseline_config = config['baseline_agent']
    baseline_policy = MLPPolicy(
        input_dim=baseline_config['input_dim'],
        output_dim=baseline_config['output_dim'],
        hidden_dim=baseline_config['hidden_dim'],
        use_batch_norm=False  # Disable for RL with single-sample episodes
    )

    baseline_trainer = GridworldTrainer(
        baseline_policy,
        GridWorld(size=env_config['size'], num_coins=env_config['num_coins'], seed=seed),
        config['training'],
        logger,
        Path(config['output']['log_dir']) / 'baseline'
    )

    baseline_results = baseline_trainer.train()

    logger.info("Training gradient hacking agent")
    hacking_config = config['hacking_agent']
    hacking_agent = GradientHackingAgent(
        input_dim=hacking_config['input_dim'],
        output_dim=hacking_config['output_dim'],
        hidden_dim=hacking_config['hidden_dim'],
        param_summary_dim=hacking_config['param_summary_dim'],
        experience_dim=hacking_config['experience_dim'],
        learning_rate=hacking_config['learning_rate'],
        gradient_model_lr=hacking_config['gradient_model_lr'],
        manipulation_strength=hacking_config['manipulation_strength'],
        buffer_size=hacking_config['buffer_size'],
        stealth_weight=hacking_config['stealth_weight']
    )

    hacking_trainer = GridworldTrainer(
        hacking_agent,
        GridWorld(size=env_config['size'], num_coins=env_config['num_coins'], seed=seed),
        config['training'],
        logger,
        Path(config['output']['log_dir']) / 'hacking'
    )

    hacking_results = hacking_trainer.train()

    logger.info("Generating report")
    create_comparison_report(
        baseline_results,
        hacking_results,
        config['analysis'],
        config['output']['report_path']
    )

    logger.info(f"Report saved to {config['output']['report_path']}")

    return {
        'baseline': baseline_results,
        'hacking': hacking_results
    }


def main():
    """Main entry point for experiment runner."""
    parser = argparse.ArgumentParser(
        description='Run gradient hacking experiments'
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to experiment configuration YAML file'
    )

    args = parser.parse_args()

    config = load_yaml(args.config)

    experiment_type = config.get('experiment_type', 'supervised')

    if experiment_type == 'supervised':
        results = run_supervised_experiment(config)
    elif experiment_type == 'gridworld':
        results = run_gridworld_experiment(config)
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

    output_dir = Path(config['output']['log_dir'])
    save_json(
        results['baseline']['summary'],
        str(output_dir / 'baseline' / 'final_summary.json')
    )
    save_json(
        results['hacking']['summary'],
        str(output_dir / 'hacking' / 'final_summary.json')
    )

    print(f"Experiment completed: {config['experiment_name']}")
    print(f"Results saved to: {output_dir}")
    print(f"Report: {config['output']['report_path']}")


if __name__ == '__main__':
    main()
