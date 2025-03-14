# Reinforcement Learning Architectures for Gin Rummy

This document explains the three reinforcement learning architectures implemented for the Gin Rummy project: REINFORCE, DQN, and MCTS.

## 1. REINFORCE Architecture

REINFORCE is a Monte Carlo policy gradient method that directly optimizes the policy without using a value function.

### Neural Network Architecture

The policy network consists of:

- **Input**: Hand matrix (4×13) and discard history
- **Convolutional layers**:
  - Conv2D: 1→32 filters (3×3 kernel)
  - Conv2D: 32→64 filters (3×3 kernel)
- **LSTM layer** for discard history:
  - Input size: 52 (card representations)
  - Hidden size: 128
- **Fully connected layers**:
  - FC1: (64×4×13 + 128) → 256
  - FC2: 256 → 128
  - FC3: 128 → 110 (action space)
- **Output**: Probability distribution over actions (softmax)

### Mathematical Foundation

#### Policy Representation
The policy π(a|s;θ) is represented by the neural network with parameters θ:

π(a|s;θ) = softmax(network_output(s;θ))

#### Objective Function
Maximize the expected return:

J(θ) = E[∑ γᵗ rₜ]

Where:
- γ is the discount factor (0.99)
- rₜ is the reward at time t

#### Policy Gradient Theorem
The policy gradient theorem gives us:

∇ᵩJ(θ) = E[∇ᵩlog π(a|s;θ) · G]

Where:
- G is the return (discounted sum of rewards)
- ∇ᵩlog π(a|s;θ) is the gradient of the log probability

#### REINFORCE Update Rule
For each episode:
1. Collect trajectory τ = {s₀, a₀, r₀, s₁, a₁, r₁, ...}
2. For each step t, calculate return Gₜ = ∑ᵏ₌ₜ γᵏ⁻ᵗ rₖ
3. Update policy parameters: θ ← θ + α∇ᵩlog π(aₜ|sₜ;θ) · Gₜ

#### Entropy Regularization
To encourage exploration:

L(θ) = -E[log π(a|s;θ) · G] - β·H[π(·|s;θ)]

Where:
- H[π(·|s;θ)] is the entropy of the policy
- β is the entropy coefficient (0.01)

### Training Process
1. Initialize policy network
2. For each episode:
   - Reset environment
   - For each step:
     - Sample action from policy
     - Take action, observe reward and next state
     - Store log probability and reward
   - Calculate returns
   - Update policy using policy gradient
3. Evaluate periodically

## 2. DQN Architecture

DQN (Deep Q-Network) is a value-based method that approximates the Q-function using a neural network.

### Neural Network Architecture

The Q-network consists of:

- **Input**: Hand matrix (4×13) and discard history
- **Convolutional layers**:
  - Conv2D: 1→32 filters (3×3 kernel)
  - Conv2D: 32→64 filters (3×3 kernel)
- **LSTM layer** for discard history:
  - Input size: 52 (card representations)
  - Hidden size: 128
- **Fully connected layers**:
  - FC1: (64×4×13 + 128) → 256
  - FC2: 256 → 128
  - FC3: 128 → 110 (action space)
- **Output**: Q-values for each action

### Mathematical Foundation

#### Q-Learning
DQN is based on Q-learning, which aims to learn the optimal action-value function:

Q*(s,a) = E[R + γ·max_a' Q*(s',a')]

#### Loss Function
The network is trained to minimize the temporal difference error:

L(θ) = E[(r + γ·max_a' Q(s',a';θ⁻) - Q(s,a;θ))²]

Where:
- θ are the parameters of the Q-network
- θ⁻ are the parameters of the target network
- r is the reward
- γ is the discount factor

#### Experience Replay
Experiences (s, a, r, s') are stored in a replay buffer and sampled randomly for training to break correlations between consecutive samples.

#### ε-Greedy Exploration
Actions are selected using an ε-greedy policy:
- With probability ε: select a random action
- With probability 1-ε: select a = argmax_a Q(s,a;θ)

The value of ε decays over time from 1.0 to 0.05.

### Training Process
1. Initialize Q-network and target network
2. Initialize replay memory
3. For each episode:
   - Reset environment
   - For each step:
     - Select action using ε-greedy policy
     - Take action, observe reward and next state
     - Store transition in replay memory
     - Sample batch from replay memory
     - Calculate target Q-values
     - Update Q-network
   - Periodically update target network
4. Evaluate periodically

## 3. MCTS Architecture

MCTS (Monte Carlo Tree Search) with neural network guidance combines tree search with policy and value networks.

### Neural Network Architecture

The policy-value network consists of:

- **Input**: Hand matrix (4×13) and discard history
- **Convolutional layers**:
  - Conv2D: 1→32 filters (3×3 kernel)
  - Conv2D: 32→64 filters (3×3 kernel)
- **LSTM layer** for discard history:
  - Input size: 52 (card representations)
  - Hidden size: 128
- **Shared representation layers**:
  - FC1: (64×4×13 + 128) → 256
  - FC2: 256 → 128
- **Policy head**:
  - FC: 128 → 110 (action space)
  - Softmax activation
- **Value head**:
  - FC: 128 → 1
  - Tanh activation

### Mathematical Foundation

#### MCTS Algorithm
MCTS consists of four phases:
1. **Selection**: Starting from the root, select child nodes according to the UCB formula until a leaf node is reached
2. **Expansion**: Add one or more child nodes to the leaf node
3. **Simulation**: Perform a rollout from the new node(s) to estimate value
4. **Backpropagation**: Update the values of all nodes in the path from the leaf to the root

#### PUCT Formula
The selection phase uses the PUCT formula:

a* = argmax_a [Q(s,a) + c_puct · P(s,a) · √N(s) / (1 + N(s,a))]

Where:
- Q(s,a) is the mean action value
- P(s,a) is the prior probability from the policy network
- N(s) is the visit count of the state
- N(s,a) is the visit count of the state-action pair
- c_puct is the exploration constant (1.0)

#### Loss Function
The network is trained to minimize:

L(θ) = (z - v)² - π^T · log p + c||θ||²

Where:
- z is the game outcome
- v is the predicted value
- π is the MCTS action probability
- p is the predicted policy
- c is the L2 regularization constant

### Training Process
1. Initialize policy-value network
2. For each episode:
   - Reset environment
   - For each step:
     - Run MCTS simulations to get action probabilities
     - Select action based on these probabilities
     - Take action, observe next state
     - Store state and MCTS probabilities
   - At the end of the episode, assign outcome values
   - Update policy-value network
3. Evaluate periodically

## Comparison of Architectures

| Aspect | REINFORCE | DQN | MCTS |
|--------|-----------|-----|------|
| **Type** | Policy-based | Value-based | Hybrid |
| **Learning** | On-policy | Off-policy | Self-play |
| **Exploration** | Entropy regularization | ε-greedy | Tree search + PUCT |
| **Sample efficiency** | Low | Medium | High |
| **Stability** | Less stable | More stable | Most stable |
| **Computation** | Low | Medium | High |
| **Hyperparameters** | Learning rate, entropy coefficient | Learning rate, ε decay, target update | Learning rate, c_puct, simulation count |

## Implementation Details

All three architectures share similar neural network designs for processing the Gin Rummy state, but differ in how they use the network outputs and how they learn from experience:

- **REINFORCE**: Learns directly from episode returns
- **DQN**: Learns from temporal difference errors using experience replay
- **MCTS**: Combines neural network guidance with tree search

Each architecture has been implemented with reward shaping to provide intermediate feedback during training, including rewards for reducing deadwood, winning, gin, and knock actions. 