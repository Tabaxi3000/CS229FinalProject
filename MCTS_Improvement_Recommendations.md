# MCTS Implementation Improvement Recommendations

Based on our analysis of the current Monte Carlo Tree Search (MCTS) implementation for Gin Rummy, we've identified several issues that are likely contributing to suboptimal performance. Here are our specific recommendations for improvements:

## 1. Action Probability Calculation

**Issue**: The current action probability calculation in `mcts.py` doesn't properly normalize probabilities after applying the valid action mask, leading to potential sampling of invalid actions.

**Fix**:
```python
# Replace in MCTSAgent._get_action_probs method
# Current problematic code:
action_probs = action_probs * valid_action_mask
return action_probs

# Improved code:
action_probs = action_probs * valid_action_mask
# Add proper normalization
action_probs = action_probs / (action_probs.sum() + 1e-8)
return action_probs
```

## 2. Search Method Improvements

**Issue**: The current MCTS search method doesn't effectively balance exploration and exploitation, especially in the large action space of Gin Rummy.

**Fix**:
```python
# In MCTS.search method, replace:
def search(self, state: Dict) -> Dict[int, float]:
    # Current implementation...

# With:
def search(self, state: Dict) -> Dict[int, float]:
    root = MCTSNode(state)
    
    # Get policy predictions from neural network
    state_tensor = {
        'hand_matrix': state['hand_matrix'].unsqueeze(0),
        'discard_history': state['discard_history'].unsqueeze(0),
        'valid_actions_mask': state['valid_actions_mask'].unsqueeze(0)
    }
    
    with torch.no_grad():
        policy_probs, _ = self.policy_network(
            state_tensor['hand_matrix'],
            state_tensor['discard_history']
        )
    policy_probs = policy_probs.squeeze()
    
    # Apply valid actions mask
    valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
    if isinstance(valid_actions, int):
        valid_actions = [valid_actions]
    
    # Expand root with all valid actions
    root.expand(policy_probs, valid_actions)
    
    # Perform simulations
    for _ in range(self.num_simulations):
        node = root
        search_path = [node]
        
        # Selection phase - select best child until reaching a leaf node
        while node.is_expanded():
            node = node.select_child()[0]
            search_path.append(node)
        
        # Expansion and evaluation phase
        parent = search_path[-2]
        action = node.action
        
        # Get value from neural network
        with torch.no_grad():
            _, value = self.policy_network(
                state_tensor['hand_matrix'],
                state_tensor['discard_history']
            )
        value = value.item()
        
        # Backpropagation phase
        for node in reversed(search_path):
            node.update(value)
    
    # Return action probabilities proportional to visit counts
    visit_counts = {child.action: child.visit_count for child in root.children.values()}
    total_visits = sum(visit_counts.values())
    
    action_probs = torch.zeros_like(policy_probs)
    for action, count in visit_counts.items():
        action_probs[action] = count / total_visits
    
    return action_probs
```

## 3. Simulation Count Adjustment

**Issue**: The default simulation count (30) is too low for effective exploration of the game tree, especially given the large action space.

**Fix**: Increase the simulation count in the training script:

```python
# In train_mcts.py, change:
mcts_agent = MCTSAgent(policy_value_net, device, num_simulations=30, c_puct=1.0)

# To:
mcts_agent = MCTSAgent(policy_value_net, device, num_simulations=100, c_puct=1.5)
```

## 4. Reward Shaping Improvements

**Issue**: The current reward shaping parameters don't provide strong enough signals for learning effective card management.

**Fix**: Adjust the reward parameters in the training script:

```python
# In train_mcts.py, change:
env = ImprovedGinRummyEnv(reward_shaping=reward_shaping, 
                         deadwood_reward_scale=0.03,
                         win_reward=2.0,
                         gin_reward=3.0,
                         knock_reward=1.0)

# To:
env = ImprovedGinRummyEnv(reward_shaping=reward_shaping, 
                         deadwood_reward_scale=0.05,  # Increased
                         win_reward=3.0,              # Increased
                         gin_reward=4.0,              # Increased
                         knock_reward=1.5,            # Increased
                         meld_reward=0.1)             # Added explicit meld reward
```

## 5. Neural Network Architecture Improvements

**Issue**: The current neural network architecture may not be capturing the complex relationships between cards effectively.

**Fix**: Enhance the network architecture with attention mechanisms:

```python
# In mcts.py, add to PolicyValueNetwork:
class PolicyValueNetwork(nn.Module):
    def __init__(self, action_space=110):
        super(PolicyValueNetwork, self).__init__()
        
        # Convolutional layers for processing hand matrix
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # LSTM for processing discard history
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        
        # Attention mechanism for card relationships
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=4)
        
        # Fully connected layer for opponent model
        self.opponent_fc = nn.Linear(52, 64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 13 + 128 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Policy and value heads
        self.policy_head = nn.Linear(128, action_space)
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, hand_matrix, discard_history, opponent_model=None):
        # ... existing code ...
        
        # Apply attention to convolutional features
        x_reshaped = x.view(batch_size, 64, -1).permute(2, 0, 1)  # [seq_len, batch, features]
        x_attended, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_attended = x_attended.permute(1, 2, 0).view(batch_size, 64, 4, 13)
        x = x + x_attended.view(batch_size, -1)  # Residual connection
        
        # ... rest of existing code ...
```

## 6. Training Process Improvements

**Issue**: The current training process doesn't provide enough episodes for complete learning, and the opponent model is too simple.

**Fix**:
1. Increase the number of training episodes:
```python
# In train_mcts.py, change:
def train_mcts(num_episodes=5000, ...)

# To:
def train_mcts(num_episodes=10000, ...)
```

2. Implement a curriculum learning approach with progressively stronger opponents:
```python
# Add to train_mcts.py:
def get_opponent_action(state, env, episode, device, opponent_model=None):
    """Get opponent action based on training progress."""
    valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
    if isinstance(valid_actions, int):
        valid_actions = [valid_actions]
    
    # Phase 1: Random opponent (first 20% of training)
    if episode < num_episodes * 0.2:
        return np.random.choice(valid_actions)
    
    # Phase 2: Heuristic opponent (next 30% of training)
    elif episode < num_episodes * 0.5:
        # Prioritize GIN and KNOCK
        if GIN in valid_actions:
            return GIN
        elif KNOCK in valid_actions:
            return KNOCK
        
        # Prioritize drawing from discard if it helps form a meld
        if DRAW_DISCARD in valid_actions and env.discard_pile:
            top_discard = env.discard_pile[-1]
            test_hand = env.player_hands[1].copy()
            test_hand.append(top_discard)
            melds_before = len(env._find_melds(env.player_hands[1]))
            melds_after = len(env._find_melds(test_hand))
            if melds_after > melds_before:
                return DRAW_DISCARD
        
        # Otherwise random
        return np.random.choice(valid_actions)
    
    # Phase 3: Previous version of the agent (final 50% of training)
    else:
        if opponent_model is not None:
            state_device = {
                'hand_matrix': state['hand_matrix'].to(device),
                'discard_history': state['discard_history'].to(device),
                'valid_actions_mask': state['valid_actions_mask'].to(device)
            }
            
            with torch.no_grad():
                action_probs, _ = opponent_model(
                    state_device['hand_matrix'],
                    state_device['discard_history']
                )
            
            # Apply mask and sample
            action_probs = action_probs.squeeze()
            action_probs = action_probs * state_device['valid_actions_mask']
            action_probs = action_probs / (action_probs.sum() + 1e-8)
            
            # Sample from distribution
            action = torch.multinomial(action_probs, 1).item()
            return action
        else:
            return np.random.choice(valid_actions)
```

## 7. Evaluation and Debugging

**Issue**: The current evaluation doesn't provide enough insights into model behavior and failure cases.

**Fix**: Enhance the evaluation script to track more detailed metrics:

```python
# Add to evaluate_agent.py:
def analyze_action_distribution(model, env, device, num_games=20):
    """Analyze the distribution of actions taken by the agent."""
    action_counts = {
        'DRAW_STOCK': 0,
        'DRAW_DISCARD': 0,
        'DISCARD': 0,
        'KNOCK': 0,
        'GIN': 0
    }
    
    deadwood_at_knock = []
    deadwood_at_gin = []
    missed_gin_opportunities = 0
    missed_knock_opportunities = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            if env.current_player == 0:  # Agent's turn
                state_device = {
                    'hand_matrix': state['hand_matrix'].to(device),
                    'discard_history': state['discard_history'].to(device),
                    'valid_actions_mask': state['valid_actions_mask'].to(device)
                }
                
                valid_actions = torch.nonzero(state_device['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                
                # Track missed opportunities
                if GIN in valid_actions:
                    missed_gin_opportunities += 1
                if KNOCK in valid_actions:
                    missed_knock_opportunities += 1
                
                # Get action from model
                action_probs, _ = model(
                    state_device['hand_matrix'],
                    state_device['discard_history']
                )
                
                # Apply mask and sample
                action_probs = action_probs.squeeze()
                action_probs = action_probs * state_device['valid_actions_mask']
                action_probs = action_probs / (action_probs.sum() + 1e-8)
                action = torch.multinomial(action_probs, 1).item()
                
                # Track action type
                if action == DRAW_STOCK:
                    action_counts['DRAW_STOCK'] += 1
                elif action == DRAW_DISCARD:
                    action_counts['DRAW_DISCARD'] += 1
                elif DISCARD_START <= action <= DISCARD_END:
                    action_counts['DISCARD'] += 1
                elif action == KNOCK:
                    action_counts['KNOCK'] += 1
                    deadwood_at_knock.append(env._calculate_deadwood(env.player_hands[0]))
                elif action == GIN:
                    action_counts['GIN'] += 1
                    deadwood_at_gin.append(env._calculate_deadwood(env.player_hands[0]))
            else:
                # Random opponent action
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = np.random.choice(valid_actions)
            
            # Take action
            next_state, _, done, _, _ = env.step(action)
            state = next_state
    
    # Calculate statistics
    total_actions = sum(action_counts.values())
    action_percentages = {k: (v / total_actions) * 100 for k, v in action_counts.items()}
    
    gin_opportunity_utilization = 0
    if missed_gin_opportunities > 0:
        gin_opportunity_utilization = (action_counts['GIN'] / missed_gin_opportunities) * 100
    
    knock_opportunity_utilization = 0
    if missed_knock_opportunities > 0:
        knock_opportunity_utilization = (action_counts['KNOCK'] / missed_knock_opportunities) * 100
    
    avg_deadwood_at_knock = np.mean(deadwood_at_knock) if deadwood_at_knock else 0
    avg_deadwood_at_gin = np.mean(deadwood_at_gin) if deadwood_at_gin else 0
    
    # Print results
    print("\nAction Distribution Analysis:")
    for action_type, percentage in action_percentages.items():
        print(f"  {action_type}: {percentage:.2f}%")
    
    print(f"\nGin Opportunity Utilization: {gin_opportunity_utilization:.2f}%")
    print(f"Knock Opportunity Utilization: {knock_opportunity_utilization:.2f}%")
    print(f"Average Deadwood at Knock: {avg_deadwood_at_knock:.2f}")
    print(f"Average Deadwood at Gin: {avg_deadwood_at_gin:.2f}")
    
    return action_percentages, gin_opportunity_utilization, knock_opportunity_utilization
```

## 8. Implementation Timeline

1. **Week 1**: Fix action probability calculation and search method
2. **Week 2**: Implement neural network architecture improvements
3. **Week 3**: Enhance reward shaping and implement curriculum learning
4. **Week 4**: Conduct extensive evaluation and fine-tune hyperparameters

By implementing these improvements, we expect to see significant performance gains in the MCTS agent, particularly in its ability to plan ahead and make strategic decisions in the mid to late game. 