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

3. For data generation (optional):
   - Install Java 11 or higher
   - Install Maven
   - Build the Java project:
```bash
cd java/MavenProject
mvn clean install
```

## Usage

### Training Data Generation
Generate training data using the Java implementation:
```bash
cd java/MavenProject
./run_data_collection.sh
```
This will create JSON files containing game states and actions from simulated Gin Rummy games.

### Training and Evaluation
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

### Python Implementation
- `python/dqn.py`: Deep Q-Network implementation
- `python/mcts.py`: Monte Carlo Tree Search implementation
- `python/reinforce.py`: REINFORCE with Value Network implementation
- `python/train_and_evaluate.py`: Training and evaluation pipeline
- `python/improved_gin_rummy_env.py`: Gin Rummy environment
- `python/rules_based_agent.py`: Rules-based baseline agent

### Java Implementation
- `java/MavenProject/src/main/java/core/`: Core Gin Rummy game logic
- `java/MavenProject/src/main/java/collector/`: Data collection utilities
- `java/MavenProject/src/main/java/player/`: Various player implementations
- `java/MavenProject/src/main/java/module/`: Game state analysis modules

## License

MIT License 