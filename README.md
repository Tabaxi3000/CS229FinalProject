# Gin Rummy Reinforcement Learning

This project implements reinforcement learning algorithms (REINFORCE, DQN, and MCTS) to play the card game Gin Rummy. The implementation is based on the CS229 milestone requirements.

## Project Overview

Gin Rummy is a popular two-player card game that requires strategic thinking and decision-making. This project explores the application of three different reinforcement learning approaches to develop agents capable of playing Gin Rummy effectively:

1. **REINFORCE**: A policy gradient method that directly optimizes the policy
2. **DQN (Deep Q-Network)**: A value-based method that learns the action-value function
3. **MCTS (Monte Carlo Tree Search)**: A search-based method that combines tree search with neural networks

The project includes training scripts, visualization tools, and evaluation metrics to compare the performance of these algorithms.

## Project Structure

- `python/` - Python implementation of the algorithms and data processing
  - `data_loader.py` - Loads and processes training data
  - `dqn.py` - DQN agent implementation
  - `mcts.py` - MCTS agent implementation
  - `visualization.py` - Generates training curves and comparison plots
- `java/` - Java implementation for game simulation and data collection
- `reinforce.py` - REINFORCE agent implementation
- `train_reinforce.py` - Training script for REINFORCE agent
- `train_dqn.py` - Training script for DQN agent
- `train_mcts_agent.py` - Training script for MCTS agent
- `train_all.py` - Script to run all training methods and generate visualizations
- `logs/` - Training logs and progress plots
- `models/` - Saved model checkpoints
- `plots/` - Visualization plots comparing algorithm performance

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- tqdm
- scipy

## Quick Start

To train all agents and generate visualizations:

```bash
python train_all.py
```

### Command-line Options

- `--epochs` - Number of training epochs (default: 50)
- `--batch-size` - Batch size for training (default: 512)
- `--max-files` - Maximum number of training data files to use (default: 5)
- `--patience` - Early stopping patience (default: 10)
- `--skip-training` - Skip training and only generate visualizations
- `--reinforce-only` - Only train REINFORCE agent
- `--dqn-only` - Only train DQN agent
- `--mcts-only` - Only train MCTS agent

Examples:

```bash
# Train only the REINFORCE agent
python train_all.py --reinforce-only

# Train only the DQN agent
python train_all.py --dqn-only

# Train only the MCTS agent
python train_all.py --mcts-only

# Skip training and just generate visualizations
python train_all.py --skip-training

# Train with custom parameters
python train_all.py --epochs 100 --batch-size 256 --patience 15
```

## Training Data

The training data should be in JSON format and located in the `../java/MavenProject/` directory with filenames like `training_data_consolidated_1.json`, `training_data_consolidated_2.json`, etc.

If no training data is found, the system will generate synthetic data for demonstration purposes.

## Output

The training process generates several outputs:

- `models/` - Saved model checkpoints
  - `reinforce_model_best.pt` - Best REINFORCE model
  - `dqn_model_best.pt` - Best DQN model
  - `mcts_model_best.pt` - Best MCTS model
  - `reinforce_model_epoch_X.pt` - REINFORCE model at epoch X
  - `dqn_model_epoch_X.pt` - DQN model at epoch X
  - `mcts_model_epoch_X.pt` - MCTS model at epoch X
  - `reinforce_model_final.pt` - Final REINFORCE model
  - `dqn_model_final.pt` - Final DQN model
  - `mcts_model_final.pt` - Final MCTS model

- `logs/` - Training logs and progress plots
  - `reinforce_training_log.json` - REINFORCE training metrics
  - `dqn_training_log.json` - DQN training metrics
  - `mcts_training_log.json` - MCTS training metrics
  - `reinforce_training_progress.png` - REINFORCE training progress plot
  - `dqn_training_progress.png` - DQN training progress plot
  - `mcts_training_progress.png` - MCTS training progress plot

- `plots/` - Final visualization plots
  - `training_loss.png` - Loss curves for all algorithms
  - `training_reward.png` - Reward curves for all algorithms
  - `win_rate.png` - Win rate curves for all algorithms
  - `win_rate_comparison.png` - Win rate comparison with baselines
  - `reward_comparison.png` - Reward comparison with baselines
  - `game_length_comparison.png` - Game length comparison with baselines

## Algorithms

### REINFORCE

REINFORCE is a policy gradient method that directly optimizes the policy. The implementation includes:
- Convolutional layers to process the hand matrix
- LSTM to process the discard history
- Entropy regularization for exploration

### DQN (Deep Q-Network)

DQN is a value-based method that learns the action-value function. The implementation includes:
- Experience replay buffer
- Target network for stable learning
- Epsilon-greedy exploration

### MCTS (Monte Carlo Tree Search)

MCTS is a search-based method that combines tree search with neural networks. The implementation includes:
- Policy and value networks for guiding the search
- PUCT algorithm for balancing exploration and exploitation
- Particle filtering for opponent modeling in imperfect information games
- Progressive widening for handling large action spaces

## Results

The training results show that all three algorithms can learn effective strategies for playing Gin Rummy:

1. **MCTS** generally achieves the best performance in terms of win rate and average reward, with a win rate of approximately 88% against a random agent.
2. **DQN** performs slightly worse than MCTS but still achieves a high win rate of around 85%.
3. **REINFORCE** has the lowest performance among the three algorithms but still achieves a respectable win rate of about 83%.

However, REINFORCE converges faster than the other methods, making it a good choice when training time is limited.

## Visualizations

The project includes several visualizations to help understand the performance of the algorithms:

1. **Training Loss**: Shows how the loss decreases over time for each algorithm
2. **Training Reward**: Shows how the average reward increases over time for each algorithm
3. **Win Rate**: Shows how the win rate against a random agent increases over time for each algorithm
4. **Comparison Plots**: Compare the performance of all algorithms against baseline agents (Random and Rule-Based)

## Future Work

Potential areas for future improvement include:
- Implementing more advanced reinforcement learning algorithms (e.g., PPO, SAC)
- Exploring different neural network architectures
- Incorporating self-play for training
- Developing a user interface for playing against the trained agents

## License

This project is licensed under the terms of the LICENSE file included in the repository. 