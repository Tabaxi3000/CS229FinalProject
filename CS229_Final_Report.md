# Reinforcement Learning for Gin Rummy: A Comparative Study of DQN, REINFORCE, and MCTS

## Abstract

This paper presents a comparative study of three reinforcement learning algorithms—Deep Q-Network (DQN), REINFORCE (Policy Gradient), and Monte Carlo Tree Search (MCTS)—applied to the card game Gin Rummy. We investigate the effectiveness of these algorithms in learning optimal strategies for this imperfect information game. Our results indicate that while all three approaches show learning capabilities, MCTS with neural network guidance demonstrates superior performance when properly tuned. We also identify key challenges in applying reinforcement learning to card games, including reward shaping, exploration-exploitation balance, and handling large action spaces.

## 1. Introduction

Card games present unique challenges for reinforcement learning due to their imperfect information nature, large state spaces, and stochastic outcomes. Gin Rummy, a popular two-player card game, offers an excellent testbed for reinforcement learning algorithms due to its combination of strategic depth and manageable complexity.

In this project, we implement and compare three distinct reinforcement learning approaches for playing Gin Rummy:
1. **Deep Q-Network (DQN)**: A value-based method that learns to estimate the expected future rewards of actions.
2. **REINFORCE (Policy Gradient)**: A policy-based method that directly learns a policy mapping states to actions.
3. **Monte Carlo Tree Search (MCTS)**: A search-based method guided by neural networks that plans several steps ahead.

The input to our algorithms is a representation of the game state, including the player's hand, the discard pile history, and valid actions. We then use our reinforcement learning models to output a predicted action (draw from stock, draw from discard, discard a card, knock, or gin).

## 2. Related Work

Reinforcement learning for card games has been an active area of research, with several notable contributions:

**Deep Reinforcement Learning for Card Games**:
- Heinrich and Silver (2016) applied deep reinforcement learning to poker, demonstrating that neural networks can learn effective strategies for imperfect information games.
- Brown and Sandholm (2019) developed Pluribus, an AI system that defeated elite human professionals in six-player no-limit Texas hold'em poker, using a form of Monte Carlo CFR (Counterfactual Regret Minimization).

**MCTS Applications**:
- Silver et al. (2017) combined MCTS with deep neural networks in AlphaGo Zero, demonstrating superhuman performance in the game of Go without human knowledge.
- Cowling et al. (2012) applied MCTS to card games with imperfect information, introducing Information Set MCTS.

**Reward Shaping in RL**:
- Ng et al. (1999) provided theoretical foundations for potential-based reward shaping, ensuring that the optimal policy remains unchanged.
- Mnih et al. (2015) demonstrated the effectiveness of DQN for learning to play Atari games, using reward clipping to handle different reward scales.

**Card Game AI**:
- Buro et al. (2009) developed AI for the card game Skat, using a combination of Monte Carlo simulation and perfect information solvers.
- Whitehouse et al. (2011) applied evolutionary algorithms to learn strategies for the card game Dou Di Zhu.

Our work differs from previous approaches by specifically focusing on Gin Rummy and providing a direct comparison of three distinct reinforcement learning paradigms on the same environment.

## 3. Dataset and Features

As this is a reinforcement learning project, we do not use a traditional dataset. Instead, our agents learn through self-play and interaction with the environment. The environment is a custom implementation of Gin Rummy with the following components:

**State Representation**:
- **Hand Matrix**: A 4×13 binary matrix representing the player's hand, where rows correspond to suits and columns to ranks.
- **Discard History**: A sequence of 52-dimensional one-hot vectors representing the cards that have been discarded.
- **Valid Actions Mask**: A binary vector indicating which actions are legal in the current state.

**Action Space**:
- Actions 0-1: Draw from stock or discard pile
- Actions 2-53: Discard one of the cards in hand
- Actions 108-109: Knock or Gin (declare the end of the game)

**Reward Structure**:
- Win: +2.0 to +3.0 (depending on configuration)
- Gin: +3.0 to +4.0
- Knock: +1.0 to +1.5
- Deadwood reduction: +0.01 to +0.05 per point reduction (with reward shaping enabled)

During training, we generated approximately 1,000 episodes for each algorithm, with each episode consisting of a complete game of Gin Rummy. For evaluation, we used 100-200 games against random and strategic opponents.

## 4. Methods

### 4.1 Deep Q-Network (DQN)

Our DQN implementation follows the architecture proposed by Mnih et al. (2015) but adapted for card games. The network takes the state representation as input and outputs Q-values for each possible action. The optimization objective is to minimize the mean squared error between the predicted Q-values and the target Q-values:

$$L(\theta) = \mathbb{E}_{(s,a,r,s')\sim D} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]$$

where $\theta$ are the network parameters, $\theta^-$ are the target network parameters, $D$ is the replay buffer, and $\gamma$ is the discount factor.

Key components of our DQN implementation include:
- Experience replay buffer to break correlations between consecutive samples
- Target network to stabilize learning
- Epsilon-greedy exploration strategy
- Double DQN to reduce overestimation bias

### 4.2 REINFORCE (Policy Gradient)

Our REINFORCE implementation directly learns a policy $\pi(a|s;\theta)$ that maps states to action probabilities. The objective is to maximize the expected return:

$$J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} R_t \right]$$

The policy gradient is given by:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi(a|s;\theta) \cdot G_t \right]$$

where $G_t$ is the return from time step $t$.

Our implementation includes:
- A policy network that outputs action probabilities
- Baseline subtraction to reduce variance
- Entropy regularization to encourage exploration

### 4.3 Monte Carlo Tree Search (MCTS)

Our MCTS implementation combines tree search with neural network guidance, similar to AlphaZero (Silver et al., 2018). The algorithm consists of four phases:
1. **Selection**: Starting from the root, select child nodes according to the UCB formula until reaching a leaf node.
2. **Expansion**: Add one or more child nodes to the leaf node.
3. **Simulation**: Use the neural network to evaluate the leaf node.
4. **Backpropagation**: Update the values and visit counts of all nodes in the path.

The UCB formula used for selection is:

$$UCB(s,a) = Q(s,a) + c_{puct} \cdot P(s,a) \cdot \frac{\sqrt{\sum_b N(s,b)}}{1 + N(s,a)}$$

where $Q(s,a)$ is the estimated value, $P(s,a)$ is the prior probability from the policy network, $N(s,a)$ is the visit count, and $c_{puct}$ is an exploration constant.

Our implementation includes:
- A combined policy-value network
- Progressive widening for handling large branching factors
- Virtual loss to encourage thread diversity in parallel search

## 5. Experiments, Results, and Discussion

### 5.1 Experimental Setup

We trained each algorithm for 1,000 episodes with the following hyperparameters:
- DQN: learning rate = 0.0001, discount factor = 0.99, epsilon decay = 0.995
- REINFORCE: learning rate = 0.0001, entropy coefficient = 0.01
- MCTS: simulations per move = 30-50, exploration constant = 1.0-1.5

For reward shaping, we experimented with different scales for deadwood reduction (0.01-0.05), win rewards (1.0-3.0), gin rewards (1.5-4.0), and knock rewards (0.5-1.5).

### 5.2 Evaluation Metrics

We evaluated our agents using the following metrics:
- **Win Rate**: Percentage of games won against random and strategic opponents
- **Average Reward**: Average reward per game
- **Gin Rate**: Percentage of games won by gin
- **Knock Rate**: Percentage of games won by knock
- **Average Deadwood**: Average deadwood count at the end of the game

### 5.3 Results

#### 5.3.1 Performance Against Random Opponent

| Algorithm | Win Rate | Avg Reward | Gin Rate | Knock Rate | Avg Deadwood |
|-----------|----------|------------|----------|------------|--------------|
| DQN       | 62.5%    | 0.87       | 15.3%    | 47.2%      | 12.8         |
| REINFORCE | 58.2%    | 0.73       | 12.1%    | 46.1%      | 14.2         |
| MCTS      | 71.3%    | 1.24       | 22.7%    | 48.6%      | 9.7          |

#### 5.3.2 Performance Against Each Other

| Match-up          | Win Rate (1st vs 2nd) |
|-------------------|------------------------|
| DQN vs REINFORCE  | 53.8% vs 46.2%        |
| DQN vs MCTS       | 41.2% vs 58.8%        |
| REINFORCE vs MCTS | 39.5% vs 60.5%        |

#### 5.3.3 Learning Curves

The learning curves show that all three algorithms improved over time, but with different characteristics:
- DQN showed steady improvement but plateaued after about 600 episodes
- REINFORCE had higher variance in performance but continued to improve
- MCTS showed the fastest initial improvement and reached the highest performance

### 5.4 Discussion

Our experiments revealed several key insights:

1. **MCTS Superiority**: MCTS consistently outperformed both DQN and REINFORCE, likely due to its ability to plan ahead and explore the game tree more effectively. This advantage was particularly pronounced in the mid to late game, where planning becomes more critical.

2. **Reward Shaping Impact**: We found that reward shaping significantly improved learning speed and final performance for all algorithms. In particular:
   - Higher deadwood reward scale (0.05) led to faster learning of card management
   - Higher gin reward (4.0) encouraged agents to aim for gin rather than knocking
   - Balanced win and knock rewards helped develop more strategic play

3. **Exploration Challenges**: All algorithms struggled with exploration in the large action space. MCTS handled this best through its tree search mechanism, while DQN and REINFORCE required careful tuning of exploration parameters.

4. **Model Performance Issues**: Several factors contributed to suboptimal performance:
   - Insufficient training episodes (1,000 is relatively small for complex card games)
   - Neural network architecture may not fully capture the game state complexity
   - The random opponent provides limited strategic challenge for advanced learning

5. **Action Selection Biases**: We observed that:
   - DQN tended to be overly conservative, often preferring to draw from the stock
   - REINFORCE sometimes developed strong biases toward specific actions
   - MCTS showed the most balanced action selection, adapting better to different game situations

## 6. Conclusion and Future Work

In this project, we implemented and compared three reinforcement learning approaches for playing Gin Rummy. Our results demonstrate that MCTS with neural network guidance performs best, followed by DQN and then REINFORCE. The planning capability of MCTS provides a significant advantage in this strategic card game.

The performance issues we identified stem from several factors: insufficient training time, neural network architecture limitations, and challenges in exploration. These findings suggest several directions for future work:

1. **Extended Training**: Train the models for significantly more episodes (10,000+) to allow for more complete learning.

2. **Improved Neural Architectures**: Explore attention mechanisms and graph neural networks to better capture the relationships between cards.

3. **Self-Play Training**: Implement a self-play training regime where agents train against improving versions of themselves rather than random opponents.

4. **Hybrid Approaches**: Combine the strengths of different algorithms, such as using DQN for the early game and MCTS for the late game.

5. **Information Set MCTS**: Adapt Information Set MCTS specifically for the imperfect information nature of Gin Rummy.

6. **Human Evaluation**: Evaluate the agents against human players to assess their strategic capabilities from a human perspective.

With these improvements, we believe reinforcement learning agents could achieve expert-level play in Gin Rummy and potentially other card games with similar characteristics.

## 7. Contributions

- **Team Member 1**: Implemented the DQN algorithm, created the environment, and conducted initial experiments.
- **Team Member 2**: Implemented the REINFORCE algorithm, designed the reward shaping mechanism, and analyzed results.
- **Team Member 3**: Implemented the MCTS algorithm, fixed critical issues in the search method, and wrote the final report.

## 8. References

1. Mnih, V., Kavukcuoglu, K., Silver, D., et al. (2015). Human-level control through deep reinforcement learning. *Nature*, 518(7540), 529-533.

2. Silver, D., Schrittwieser, J., Simonyan, K., et al. (2017). Mastering the game of Go without human knowledge. *Nature*, 550(7676), 354-359.

3. Heinrich, J., & Silver, D. (2016). Deep reinforcement learning from self-play in imperfect-information games. *arXiv preprint arXiv:1603.01121*.

4. Brown, N., & Sandholm, T. (2019). Superhuman AI for multiplayer poker. *Science*, 365(6456), 885-890.

5. Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations: Theory and application to reward shaping. *ICML*, 99, 278-287.

6. Cowling, P. I., Powley, E. J., & Whitehouse, D. (2012). Information set monte carlo tree search. *IEEE Transactions on Computational Intelligence and AI in Games*, 4(2), 120-143.

7. Buro, M., Long, J. R., Furtak, T., & Sturtevant, N. R. (2009). Improving state evaluation, inference, and search in trick-based card games. *IJCAI*, 9, 1407-1413.

8. Whitehouse, D., Powley, E. J., & Cowling, P. I. (2011). Determinization and information set Monte Carlo tree search for the card game Dou Di Zhu. *IEEE Conference on Computational Intelligence and Games (CIG)*, 87-94. 