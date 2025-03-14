# Gin Rummy Reinforcement Learning

This project implements three reinforcement learning approaches (REINFORCE, DQN, and MCTS) for the card game Gin Rummy.

## Project Overview

Gin Rummy is a strategic card game where players aim to form melds (sets or runs) with their cards. This project explores how different reinforcement learning algorithms can learn to play this game effectively.

The three implemented approaches are:
- **REINFORCE**: A policy gradient method that directly optimizes the policy.
- **DQN (Deep Q-Network)**: A value-based method that learns the action-value function.
- **MCTS (Monte Carlo Tree Search)**: A search-based method that plans ahead by simulating possible future game states.

## Project Structure

- `python/`: Contains the Python implementation of the algorithms
  - `reinforce.py`: Implementation of the REINFORCE algorithm
  - `dqn.py`: Implementation of the Deep Q-Network algorithm
  - `mcts.py`: Implementation of the Monte Carlo Tree Search algorithm
  - `train_*.py`: Training scripts for each algorithm
  - Other utility scripts for data processing and evaluation
- `logs/`: Contains training logs and progress plots
- `models/`: Contains trained model checkpoints
- `plots/`: Contains visualization plots of results

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- Matplotlib
- Pandas
- tqdm
- scipy

## Quick Start

To train all agents:

```bash
python train_all.py
```

Command-line options:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate
- `--gamma`: Discount factor
- `--device`: Device to use (cpu or cuda)

## Training Data

The training process generates data that is used to train the models. The data includes:
- Game states
- Actions taken
- Rewards received
- Next states
- Terminal flags

## Output

The training process produces:
- Trained models saved in the `models/` directory
- Training logs saved in the `logs/` directory
- Visualization plots saved in the `plots/` directory

## Algorithms

### REINFORCE

REINFORCE is a policy gradient method that directly optimizes the policy. It uses the entire episode to update the policy parameters.

### DQN

DQN is a value-based method that learns the action-value function. It uses experience replay and target networks to stabilize training.

### MCTS

MCTS is a search-based method that plans ahead by simulating possible future game states. It balances exploration and exploitation using the UCB formula.

## Results

The results show that:
- MCTS achieves the highest win rate against random opponents
- REINFORCE converges faster than DQN
- DQN achieves more stable performance over time

## Visualizations

The project includes several visualizations:
- Training progress (loss and reward)
- Win rate comparison between algorithms
- Game length comparison
- Reward comparison

## Future Work

Potential improvements and extensions:
- Implement more advanced algorithms (A2C, PPO, etc.)
- Explore different neural network architectures
- Implement self-play training
- Add human vs. AI gameplay interface

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Repository Update

This repository has been updated on March 13, 2024 with the latest code and visualizations.

## GitHub CLI

This repository can be cloned using GitHub CLI with the following command:

```bash
gh repo clone Tabaxi3000/PrivateCS229
``` 