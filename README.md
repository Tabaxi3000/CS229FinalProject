# Gin Rummy AI Agents

This repository contains implementations of various reinforcement learning agents for playing Gin Rummy:

- Deep Q-Network (DQN)
- Monte Carlo Tree Search (MCTS)
- REINFORCE with Value Network
- Rules-Based Agent (baseline)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Tabaxi3000/PrivateCS229.git
cd PrivateCS229
```

2. Install dependencies:
```bash
pip install -r python/requirements.txt
```

## Usage

Train and evaluate all agents:
```bash
python python/train_and_evaluate.py --train
```

The script will:
- Train DQN, MCTS, and REINFORCE agents in parallel
- Save models and metrics periodically
- Generate evaluation plots
- Compare performance against a rules-based agent

## Project Structure

- `python/dqn.py`: Deep Q-Network implementation
- `python/mcts.py`: Monte Carlo Tree Search implementation
- `python/reinforce.py`: REINFORCE with Value Network implementation
- `python/train_and_evaluate.py`: Training and evaluation pipeline
- `python/improved_gin_rummy_env.py`: Gin Rummy environment
- `python/rules_based_agent.py`: Rules-based baseline agent

## License

MIT License 