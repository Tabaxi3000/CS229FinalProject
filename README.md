# Gin Rummy Reinforcement Learning

This repository contains the implementation of various reinforcement learning agents for playing the card game Gin Rummy, developed as part of the CS229 Machine Learning course project.

## Project Overview

Gin Rummy is a two-player card game that involves both strategic decision-making and imperfect information, making it an interesting challenge for reinforcement learning. This project implements and compares several reinforcement learning approaches:

- Deep Q-Network (DQN)
- REINFORCE (Policy Gradient)
- Monte Carlo Tree Search (MCTS)
- Improved Monte Carlo Tree Search (with neural network policy and value functions)

## Repository Structure

```
.
├── models/                  # Trained model files
├── python/                  # Python source code
│   ├── agents/              # Agent implementations
│   │   ├── dqn_agent.py     # DQN agent implementation
│   │   ├── reinforce_agent.py  # REINFORCE agent implementation
│   │   ├── mcts_agent.py    # MCTS agent implementation
│   │   └── policy_value_net.py  # Neural network for MCTS
│   ├── environment/         # Gin Rummy environment
│   ├── train_for_cs229.py   # Training script for CS229 milestone
│   ├── generate_cs229_graphs.py  # Script to generate visualizations
│   └── next_steps.py        # Future work and improvements
├── cs229_win_rate_comparison.png  # Win rate comparison visualization
├── cs229_learning_curves.png      # Learning curves visualization
└── other visualizations...
```

## Results

The project compares the performance of different reinforcement learning approaches for playing Gin Rummy. Key findings include:

- DQN achieves a win rate of approximately 43% against a random opponent
- REINFORCE achieves a win rate of approximately 39% against a random opponent
- MCTS achieves a win rate of approximately 65% against a random opponent
- Improved MCTS achieves a win rate of approximately 72% against a random opponent

Detailed visualizations of training metrics and performance comparisons can be found in the generated PNG files.

## Future Work

The `next_steps.py` script outlines planned improvements to the project, including:

1. Implementing self-play training
2. Adding opponent modeling capabilities
3. Optimizing hyperparameters
4. Implementing ensemble methods
5. Evaluating against human players

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- PyTorch (for neural network-based agents)

## Usage

To train the agents:

```bash
python python/train_for_cs229.py --episodes 1000
```

To generate visualizations:

```bash
python python/generate_cs229_graphs.py --metrics-file cs229_training_metrics.json --save-dir .
```

To view the next steps and projected improvements:

```bash
python python/next_steps.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CS229 course staff for guidance and support
- OpenAI for reinforcement learning resources
- DeepMind for MCTS and AlphaZero inspiration 