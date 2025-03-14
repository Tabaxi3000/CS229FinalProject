# Gin Rummy RL Training Guide

This guide explains how to train and evaluate the reinforcement learning models for Gin Rummy.

## Prerequisites

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Training Scripts

### Quick Start

For a quick test of the training pipeline, run:

```bash
./quick_train.sh
```

This will train all three models (DQN, REINFORCE, and MCTS) for 100 episodes each with evaluation every 20 episodes.

### Full Training

To run a full training session with all models:

```bash
./train_all_models.py --episodes 1000 --eval-interval 100 --save-interval 200 --models all
```

### Training Individual Models

You can train specific models by specifying the `--models` parameter:

```bash
# Train only DQN
./train_all_models.py --episodes 1000 --models dqn

# Train only REINFORCE
./train_all_models.py --episodes 1000 --models reinforce

# Train only MCTS
./train_all_models.py --episodes 1000 --models mcts
```

### Command-Line Arguments

- `--episodes`: Number of episodes to train each model (default: 1000)
- `--eval-interval`: Interval for evaluation during training (default: 100)
- `--save-interval`: Interval for saving models during training (default: 200)
- `--models`: Models to train: 'dqn', 'reinforce', 'mcts', or 'all' (default: 'all')
- `--plot-only`: Only plot existing metrics without training

## Output Files

The training scripts will create the following directories:

- `models/`: Contains saved model weights
- `metrics/`: Contains JSON files with training metrics
- `plots/`: Contains learning curve plots

## Learning Curves

After training, you can view the learning curves in the `plots/` directory:

- `learning_curves.png`: Combined learning curves for all trained models
- `win_rate_comparison.png`: Comparison of win rates across models
- `dqn_training_metrics.png`: Detailed metrics for DQN training
- `reinforce_training_metrics.png`: Detailed metrics for REINFORCE training
- `mcts_training_metrics.png`: Detailed metrics for MCTS training

## Generating Only Plots

If you've already trained models and just want to generate plots from the saved metrics:

```bash
./train_all_models.py --plot-only
```

## Troubleshooting

If you encounter any issues:

1. Make sure all required Python packages are installed
2. Check that the Python scripts are in the correct directory structure
3. Ensure the scripts have executable permissions (`chmod +x script_name.py`)
4. Check the console output for specific error messages

For PyTorch-specific errors related to MKL or CUDA, you may need to reinstall PyTorch with the appropriate configuration for your system. 