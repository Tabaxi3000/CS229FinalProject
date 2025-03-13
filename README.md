# Algorithm of Reinforcement Learning for Imperfect Information Card Game: Gin Rummy

Authors: Caleb Choe, Mason Eyler, Leo Li

## Project Overview

This project implements multiple reinforcement learning approaches to play Gin Rummy, featuring a robust data collection pipeline and various RL algorithms. The implementation builds upon existing Gin Rummy game logic to create a comprehensive reinforcement learning framework.

### Recent Updates (March 2025)

#### Data Collection and Training Pipeline

1. **Data Collection Infrastructure**
   - Parallel data collection using Java threads
   - Generates 100,000 games worth of training data
   - Data includes game states, actions, rewards, and outcomes
   - Automatically splits data into 10 JSON files (10,000 games each)
   - Files stored in `python/data/` directory

2. **Reinforcement Learning Implementations**
   - Deep Q-Network (DQN) with experience replay
   - REINFORCE (Policy Gradient)
   - Monte Carlo Tree Search (MCTS) with neural network policy
   - Custom neural architectures for card game state representation
   - Efficient data loading and preprocessing utilities

3. **Project Structure**
   ```
   .
   ├── java/                    # Java implementation
   │   └── MavenProject/       # Core game engine and data collection
   ├── python/                 # Python RL implementations
   │   ├── data/              # Training data (gitignored)
   │   ├── dqn.py            # DQN implementation
   │   ├── reinforce.py      # REINFORCE implementation
   │   ├── mcts.py           # MCTS implementation
   │   └── data_loader.py    # Data loading utilities
   └── README.md
   ```

4. **Getting Started**
   - Install Java dependencies: `cd java/MavenProject && mvn install`
   - Install Python dependencies: `pip install -r requirements.txt`
   - Collect training data: `cd java/MavenProject && mvn exec:java -Dexec.mainClass="collector.DataCollectionMain"`
   - Training data will be saved in `python/data/`

Note: Training data files are not included in the repository due to size. You'll need to generate them using the data collection pipeline.

## Features

### 1. Data Collection Pipeline
- Parallel data collection using Java threads
- Configurable number of games and threads
- Efficient JSON data format
- Automatic file splitting for large datasets

### 2. Reinforcement Learning Implementations

#### Deep Q-Network (DQN)
- Experience replay buffer
- Target network for stable learning
- Custom neural architecture for card games
- Epsilon-greedy exploration

#### REINFORCE (Policy Gradient)
- Direct policy optimization
- Entropy regularization
- Custom reward shaping
- State value estimation

#### Monte Carlo Tree Search (MCTS)
- Neural network policy guidance
- Progressive widening
- Particle filtering for opponent modeling
- TD(λ) backpropagation

### 3. State Representation
- Hand matrix representation (4x13 for suits/ranks)
- Discard history encoding
- Opponent modeling
- Action masking for legal moves

## Getting Started

### Prerequisites
- Java 11 or higher
- Maven
- Python 3.8 or higher
- PyTorch

### Installation

1. Install Java dependencies:
```bash
cd java/MavenProject
mvn install
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

### Data Collection

Run the data collection process:
```bash
cd java/MavenProject
mvn exec:java -Dexec.mainClass="collector.DataCollectionMain"
```

This will generate training data in `python/data/` directory.

### Training Models

Each algorithm can be trained independently:

```bash
# Train DQN
python python/dqn.py

# Train REINFORCE
python python/reinforce.py

# Train MCTS
python python/mcts.py
```

## Notes

- Training data files are not included in the repository due to size
- The data collection process automatically splits data into manageable chunks
- Models automatically save checkpoints during training

## Authors

- Caleb Choe
- Mason Eyler
- Leo Li

## License

This project is licensed under the GNU General Public License v2.0 - see the LICENSE file for details.
