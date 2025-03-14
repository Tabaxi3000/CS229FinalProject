# Gin Rummy Algorithm

This repository contains implementations of various reinforcement learning algorithms for playing Gin Rummy:

- Deep Q-Network (DQN)
- REINFORCE (Policy Gradient)
- Monte Carlo Tree Search (MCTS)

## Recent Fixes

- Fixed MCTS agent action probability calculation and search method
- Fixed action sampling in train_mcts.py to ensure valid actions
- Updated train_mcts.py to handle 5-value return from env.step()

## Requirements

- Python 3.7+
- PyTorch
- NumPy
- tqdm

## Usage

To train a model:

```bash
cd python
python train_mcts.py --episodes 1000 --reward-shaping --deadwood-reward-scale 0.03 --win-reward 2.0 --gin-reward 3.0 --knock-reward 1.0 --eval-interval 200 --save-interval 500 --simulations 30 --save
```

## Note

Large data files and model files are excluded from this repository. You'll need to generate your own training data. 