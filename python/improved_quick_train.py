#!/usr/bin/env python3

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import math
import copy
import matplotlib.pyplot as plt

# Import our environment
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN

# Constants
BATCH_SIZE = 64
GAMMA = 0.99
DQN_LR = 0.001
REINFORCE_LR = 0.0005
MCTS_LR = 0.0002
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
DQN_EPISODES = 200
REINFORCE_EPISODES = 200
MCTS_EPISODES = 200
EVAL_INTERVAL = 20
SAVE_INTERVAL = 50

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# Constants for quick training
QUICK_EPISODES = 1000
EPSILON_START = 0.95
EPSILON_END = 0.1
EPSILON_DECAY = 15000
REWARD_SCALING = 2.0

# Constants for REINFORCE training
REINFORCE_GAMMA = 0.99

# Constants for MCTS training
MCTS_SIMULATIONS = 50
MCTS_C_PUCT = 1.0

# Define action priorities - higher values mean higher priority
ACTION_PRIORITIES = {
    GIN: 1000,    # Highest priority
    KNOCK: 500    # Second highest priority
}

# Prioritized Experience Replay with reduced prioritization effect
class PrioritizedReplayMemory:
    """Prioritized Replay Memory for DQN training"""
    def __init__(self, capacity, alpha=0.4, beta_start=0.6, beta_end=1.0, beta_frames=10000):
        self.capacity = capacity
        self.alpha = alpha  # Reduced from 0.6 to 0.4 to decrease prioritization effect
        self.beta_start = beta_start  # Increased from 0.4 to 0.6 for better bias correction
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.frame = 1  # For beta calculation
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        
    def beta_by_frame(self, frame_idx):
        """Calculate beta based on current frame"""
        return min(self.beta_end, self.beta_start + frame_idx * (self.beta_end - self.beta_start) / self.beta_frames)
        
    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        max_priority = self.priorities.max() if self.memory else 1.0
        
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        """Sample a batch of transitions with priorities"""
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
            
        # Calculate sampling probabilities
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        
        # Calculate importance sampling weights
        beta = self.beta_by_frame(self.frame)
        self.frame += 1
        
        # Calculate weights
        weights = (len(self.memory) * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        
        samples = [self.memory[idx] for idx in indices]
        return samples, indices, weights
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled transitions"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.memory)

def quick_train_dqn(env, device):
    """Train a DQN agent for Gin Rummy."""
    # Create networks
    policy_net = DQNetwork().to(device)
    target_net = DQNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Create optimizer
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.0005)
    
    # Create replay memory
    memory = PrioritizedReplayMemory(10000)
    
    # Training parameters
    batch_size = 64
    gamma = 0.99
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 1000
    target_update = 10
    num_episodes = 1000
    
    # Tracking variables
    rewards = []
    shaped_rewards = []
    losses = []
    
    # Progress bar
    progress_bar = tqdm(range(num_episodes), desc="DQN Training")
    
    for i_episode in progress_bar:
        # Initialize the environment and state
        state = env.reset()
        episode_reward = 0
        episode_shaped_reward = 0
        episode_loss = 0
        
        # Calculate epsilon for this episode
        eps_threshold = eps_end + (eps_start - eps_end) * \
            math.exp(-1. * i_episode / eps_decay)
        
        done = False
        while not done:
            # Select and perform an action
            if env.current_player == 0:  # Our agent's turn
                # Move state to device
                state_device = {
                    'hand_matrix': state['hand_matrix'].to(device),
                    'discard_history': state['discard_history'].to(device),
                    'valid_actions_mask': state['valid_actions_mask'].to(device)
                }
                
                # Get valid actions
                valid_actions = torch.nonzero(state_device['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                
                # Prioritize GIN and KNOCK actions if available
                if GIN in valid_actions:
                    action = GIN
                elif KNOCK in valid_actions:
                    action = KNOCK
                else:
                    # Epsilon-greedy action selection
                    if random.random() < eps_threshold:
                        action = random.choice(valid_actions)
                    else:
                        with torch.no_grad():
                            q_values = policy_net(
                                state_device['hand_matrix'],
                                state_device['discard_history']
                            )
                            
                            # Apply mask to Q-values
                            masked_q_values = q_values.squeeze().clone()
                            masked_q_values[~state_device['valid_actions_mask'].bool()] = float('-inf')
                            action = masked_q_values.argmax().item()
                
                # Take action
                next_state, reward, done, _, info = env.step(action)
                
                # Store original and shaped rewards
                original_reward = info.get('original_reward', 0)
                shaped_reward = info.get('shaped_reward', reward)
                
                # Update episode rewards
                episode_reward += original_reward
                episode_shaped_reward += shaped_reward
                
                # Store transition in memory
                next_state_device = {
                    'hand_matrix': torch.tensor(next_state['hand_matrix'], device=device),
                    'discard_history': torch.tensor(next_state['discard_history'], device=device),
                    'valid_actions_mask': torch.tensor(next_state['valid_actions_mask'], device=device)
                }
                
                # Calculate TD error for prioritized replay
                with torch.no_grad():
                    # Current Q-value
                    current_q = policy_net(
                        state_device['hand_matrix'],
                        state_device['discard_history']
                    )[0, action]
                    
                    # Next Q-value (max over valid actions)
                    next_q_values = target_net(
                        next_state_device['hand_matrix'],
                        next_state_device['discard_history']
                    )[0]
                    next_q_values[~next_state_device['valid_actions_mask'].bool()] = float('-inf')
                    next_q = next_q_values.max()
                    
                    # Target Q-value
                    target_q = shaped_reward + (gamma * next_q * (1 - done))
                    
                    # TD error
                    td_error = abs(current_q - target_q).item()
                
                # Store transition with priority
                memory.push(
                    state,
                    action,
                    shaped_reward,
                    next_state,
                    done,
                    td_error
                )
                
                # Update state
                state = next_state
                
                # Perform one step of optimization
                if len(memory) >= batch_size:
                    # Sample batch
                    batch, indices, weights = memory.sample(batch_size)
                    
                    # Unpack batch
                    batch_state = [s for s, _, _, _, _ in batch]
                    batch_action = torch.tensor([a for _, a, _, _, _ in batch], device=device)
                    batch_reward = torch.tensor([r for _, _, r, _, _ in batch], device=device, dtype=torch.float)
                    batch_next_state = [s for _, _, _, s, _ in batch]
                    batch_done = torch.tensor([d for _, _, _, _, d in batch], device=device, dtype=torch.float)
                    
                    # Convert states to tensors
                    batch_state_hand = torch.cat([torch.tensor(s['hand_matrix'], device=device) for s in batch_state])
                    batch_state_discard = torch.cat([torch.tensor(s['discard_history'], device=device) for s in batch_state])
                    batch_next_state_hand = torch.cat([torch.tensor(s['hand_matrix'], device=device) for s in batch_next_state])
                    batch_next_state_discard = torch.cat([torch.tensor(s['discard_history'], device=device) for s in batch_next_state])
                    batch_next_state_mask = torch.stack([torch.tensor(s['valid_actions_mask'], device=device) for s in batch_next_state])
                    
                    # Compute Q(s_t, a)
                    q_values = policy_net(batch_state_hand, batch_state_discard)
                    state_action_values = q_values.gather(1, batch_action.unsqueeze(1)).squeeze(1)
                    
                    # Compute V(s_{t+1}) for all next states
                    with torch.no_grad():
                        next_q_values = target_net(batch_next_state_hand, batch_next_state_discard)
                        # Apply mask to next Q-values
                        next_q_values[~batch_next_state_mask.bool()] = float('-inf')
                        next_state_values = next_q_values.max(1)[0]
                        # Compute the expected Q values
                        expected_state_action_values = batch_reward + (gamma * next_state_values * (1 - batch_done))
                    
                    # Compute loss
                    weights_tensor = torch.tensor(weights, device=device, dtype=torch.float)
                    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
                    loss = (loss * weights_tensor).mean()
                    
                    # Update priorities
                    td_errors = torch.abs(state_action_values - expected_state_action_values).detach().cpu().numpy()
                    memory.update_priorities(indices, td_errors)
                    
                    # Optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                    optimizer.step()
                    
                    episode_loss += loss.item()
            
            else:  # Random opponent's turn
                # Get valid actions
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                
                # Select random action
                action = random.choice(valid_actions)
                
                # Take action
                next_state, _, done, _, _ = env.step(action)
                
                # Update state
                state = next_state
        
        # Update target network
        if i_episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Store episode metrics
        rewards.append(episode_reward)
        shaped_rewards.append(episode_shaped_reward)
        losses.append(episode_loss)
        
        # Update progress bar
        progress_bar.set_postfix({
            'epsilon': f"{eps_threshold:.2f}",
            'loss': f"{episode_loss:.2f}",
            'original': f"{episode_reward:.2f}",
            'shaped': f"{episode_shaped_reward:.2f}"
        })
        
        # Save model periodically
        if (i_episode + 1) % 100 == 0:
            torch.save(policy_net.state_dict(), f"models/improved_dqn_{i_episode + 1}.pt")
    
    # Save final model
    torch.save(policy_net.state_dict(), "models/improved_dqn_final.pt")
    
    # Plot training metrics
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(rewards)
    plt.title('Original Rewards')
    plt.subplot(1, 3, 2)
    plt.plot(shaped_rewards)
    plt.title('Shaped Rewards')
    plt.subplot(1, 3, 3)
    plt.plot(losses)
    plt.title('Losses')
    plt.tight_layout()
    plt.savefig('dqn_training_metrics.png')
    
    return policy_net

def evaluate_dqn(policy_net, device, env, num_games=20):
    """Evaluate a DQN agent."""
    wins = 0
    total_reward = 0
    gin_opportunities = 0
    knock_opportunities = 0
    gin_taken = 0
    knock_taken = 0
    
    for game in range(num_games):
        state = env.reset()
        done = False
        game_reward = 0
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                # Move state to device
                state_device = {
                    'hand_matrix': state['hand_matrix'].to(device),
                    'discard_history': state['discard_history'].to(device),
                    'valid_actions_mask': state['valid_actions_mask'].to(device)
                }
                
                # Get valid actions
                valid_actions = torch.nonzero(state_device['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                
                # Track GIN and KNOCK opportunities
                if GIN in valid_actions:
                    gin_opportunities += 1
                if KNOCK in valid_actions:
                    knock_opportunities += 1
                
                # Get action from model
                with torch.no_grad():
                    q_values = policy_net(
                        state_device['hand_matrix'],
                        state_device['discard_history']
                    )
                    
                    # Prioritize GIN and KNOCK actions
                    if GIN in valid_actions:
                        action = GIN
                        gin_taken += 1
                    elif KNOCK in valid_actions:
                        action = KNOCK
                        knock_taken += 1
                    else:
                        # Apply mask to Q-values
                        masked_q_values = q_values.squeeze().clone()
                        masked_q_values[~state_device['valid_actions_mask'].bool()] = float('-inf')
                        action = masked_q_values.argmax().item()
            
            # Random opponent's turn
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Track rewards for player 0 (our agent)
            if env.current_player == 1:  # Just took action as player 0
                game_reward += reward
                
                # Print detailed results for each game
                if done and 'outcome' in info:
                    if info['outcome'] == 'win' or info['outcome'] == 'gin':
                        wins += 1
                        print(f"Game {game+1}: Agent WON! Reward: {game_reward:.2f}, Shaped: {info.get('shaped_reward', 'N/A')}")
                    else:
                        print(f"Game {game+1}: Agent lost. Reward: {game_reward:.2f}, Shaped: {info.get('shaped_reward', 'N/A')}")
            
            # Update state
            state = next_state
        
        total_reward += game_reward
    
    # Print action statistics
    print("\nEvaluation Results:")
    print(f"  Win rate: {wins/num_games:.2f} ({wins}/{num_games})")
    print(f"  Average reward: {total_reward/num_games:.2f}")
    
    if gin_opportunities > 0:
        print(f"  GIN opportunities: {gin_opportunities}, taken: {gin_taken} ({gin_taken/gin_opportunities*100:.1f}%)")
    if knock_opportunities > 0:
        print(f"  KNOCK opportunities: {knock_opportunities}, taken: {knock_taken} ({knock_taken/knock_opportunities*100:.1f}%)")
    
    return wins / num_games, total_reward / num_games

def quick_train_reinforce(env, device):
    """Quick training for REINFORCE agent with improved environment and reward shaping"""
    print("Starting quick REINFORCE training with reward shaping...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Modify the environment to give higher rewards for winning actions
    env.win_reward = 20.0   # Doubled from 10.0
    env.gin_reward = 40.0   # Doubled from 20.0
    env.knock_reward = 10.0  # Doubled from 5.0
    
    print(f"Environment constants - GIN: {GIN}, KNOCK: {KNOCK}")
    print(f"Reward structure - win: {env.win_reward}, gin: {env.gin_reward}, knock: {env.knock_reward}")
    
    # Initialize agent
    policy = PolicyNetwork().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=REINFORCE_LR)
    
    # Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.25)
    
    # Training metrics
    rewards = []
    shaped_rewards = []
    original_rewards = []
    losses = []
    win_rates = []
    
    # Diagnostic counters
    gin_opportunities = 0
    knock_opportunities = 0
    gin_taken = 0
    knock_taken = 0
    
    # Action statistics
    action_counts = {}
    
    # Training loop with progress bar
    print(f"Training REINFORCE for {REINFORCE_EPISODES} episodes...")
    episode_pbar = tqdm(range(1, REINFORCE_EPISODES + 1), desc="REINFORCE Training")
    
    for episode in episode_pbar:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_shaped_reward = 0
        episode_original_reward = 0
        done = False
        
        # Storage for episode
        states = []
        actions = []
        rewards_list = []
        original_rewards_list = []
        log_probs = []
        
        # Play one episode
        while not done:
            # Move state to device
            state_device = {
                'hand_matrix': state['hand_matrix'].to(device),
                'discard_history': state['discard_history'].to(device),
                'valid_actions_mask': state['valid_actions_mask'].to(device)
            }
            
            # Get valid actions
            valid_actions = torch.nonzero(state_device['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
            
            # Track GIN and KNOCK opportunities
            if GIN in valid_actions:
                gin_opportunities += 1
            if KNOCK in valid_actions:
                knock_opportunities += 1
            
            # Forward pass through policy network
            action_probs = policy(
                state_device['hand_matrix'],
                state_device['discard_history']
            )
            
            # Apply mask to action probabilities
            action_probs = action_probs.squeeze()
            action_probs = action_probs * state_device['valid_actions_mask']
            
            # Boost probabilities for GIN and KNOCK actions
            if GIN in valid_actions:
                action_probs[GIN] *= 10.0  # Strongly boost GIN probability
            if KNOCK in valid_actions:
                action_probs[KNOCK] *= 5.0  # Moderately boost KNOCK probability
            
            # Normalize probabilities
            action_probs = action_probs / (action_probs.sum() + 1e-8)
            
            # Determine exploration rate based on episode progress
            exploration_rate = max(0.1, 0.5 - 0.01 * episode)  # Decaying exploration rate
            
            # Prioritize GIN and KNOCK actions if they're valid
            if GIN in valid_actions and (random.random() < 0.99 or episode > REINFORCE_EPISODES // 2):
                # Very high chance to take GIN, especially in later episodes
                action = GIN
                gin_taken += 1
                print(f"GIN action taken in episode {episode}!")
            elif KNOCK in valid_actions and (random.random() < 0.95 or episode > REINFORCE_EPISODES // 2):
                # High chance to take KNOCK, especially in later episodes
                action = KNOCK
                knock_taken += 1
                print(f"KNOCK action taken in episode {episode}!")
            else:
                # Use epsilon-greedy for exploration
                if random.random() < exploration_rate:
                    action = random.choice(valid_actions)
                else:
                    # Sample action from policy
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample().item()
            
            # Track action statistics
            action_counts[action] = action_counts.get(action, 0) + 1
            
            # Calculate log probability of action
            log_prob = torch.log(action_probs[action] + 1e-8)
            
            # Take action
            next_state, reward, done, truncated, info = env.step(action)
            
            # Track rewards
            episode_reward += reward
            episode_shaped_reward += reward
            episode_original_reward += info['original_reward']
            
            # Check if we won the game
            if done and info.get('outcome') in ['win', 'gin'] and reward > 0:
                print(f"WON GAME with {info['outcome']} in episode {episode}! Reward: {reward}")
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards_list.append(reward)
            original_rewards_list.append(info['original_reward'])
            log_probs.append(log_prob)
            
            # Move to next state
            state = next_state
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards_list):
            G = r + GAMMA * G
            returns.insert(0, G)
        returns = torch.tensor(returns, device=device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()
        
        # Step the learning rate scheduler
        scheduler.step()
        
        # Track metrics
        rewards.append(episode_reward)
        shaped_rewards.append(episode_shaped_reward)
        original_rewards.append(episode_original_reward)
        losses.append(policy_loss.item())
        
        # Print diagnostic info every 10 episodes
        if episode % 10 == 0:
            print(f"\nDiagnostic info after episode {episode}:")
            if gin_opportunities > 0:
                print(f"GIN opportunities: {gin_opportunities}, taken: {gin_taken} ({gin_taken/gin_opportunities*100:.1f}%)")
            if knock_opportunities > 0:
                print(f"KNOCK opportunities: {knock_opportunities}, taken: {knock_taken} ({knock_taken/knock_opportunities*100:.1f}%)")
        
        # Evaluate agent periodically
        if episode % EVAL_INTERVAL == 0:
            win_rate, avg_shaped_reward, avg_original_reward = evaluate_reinforce(policy, device, env, num_games=10)
            win_rates.append(win_rate)
            print(f"Action statistics: {action_counts}")
            print(f"Episode {episode}/{REINFORCE_EPISODES} - Win rate: {win_rate:.2f}, Avg shaped reward: {avg_shaped_reward:.2f}, Avg original reward: {avg_original_reward:.2f}")
            print(f"Current learning rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model based on win rate
            if win_rate > 0 or episode % SAVE_INTERVAL == 0:
                torch.save(
                    policy.state_dict(),
                    f"models/improved_reinforce_episode_{episode}_winrate_{win_rate:.2f}.pt"
                )
        
        # Update progress bar
        avg_reward = sum(rewards[-min(10, len(rewards)):]) / min(10, len(rewards))
        avg_original = sum(original_rewards[-min(10, len(original_rewards)):]) / min(10, len(original_rewards))
        avg_loss = sum(losses[-min(10, len(losses)):]) / min(10, len(losses)) if losses else 0
        episode_pbar.set_postfix(
            loss=f"{avg_loss:.4f}",
            original=f"{avg_original:.2f}",
            shaped=f"{avg_reward:.2f}"
        )
    
    # Save final model
    torch.save(
        policy.state_dict(),
        "models/improved_reinforce_final.pt"
    )
    
    print("Quick REINFORCE training complete!")
    return policy

def evaluate_reinforce(policy, device, env, num_games=10):
    """Evaluate the REINFORCE policy on a number of games"""
    wins = 0
    total_shaped_reward = 0
    total_original_reward = 0
    action_stats = {}
    
    # Constants for GIN and KNOCK actions
    GIN = env.GIN
    KNOCK = env.KNOCK
    
    # Track GIN and KNOCK opportunities and actions
    gin_opportunities = 0
    gin_taken = 0
    knock_opportunities = 0
    knock_taken = 0
    
    print("===== DETAILED EVALUATION RESULTS =====")
    
    for game in range(num_games):
        print(f"Evaluating game {game+1}/{num_games}...")
        state = env.reset()
        done = False
        game_shaped_reward = 0
        game_original_reward = 0
        
        while not done:
            # Check for GIN and KNOCK opportunities
            if env.current_player == 0:
                deadwood_count = env.get_deadwood_count(0)
                if deadwood_count == 0:
                    gin_opportunities += 1
                    print(f"  Game {game+1}: GIN opportunity detected!")
                elif deadwood_count <= 10:
                    knock_opportunities += 1
                    print(f"  Game {game+1}: KNOCK opportunity detected!")
            
            if env.current_player == 0:
                # Move state to device
                state_device = {
                    'hand_matrix': state['hand_matrix'].to(device),
                    'discard_history': state['discard_history'].to(device),
                    'valid_actions_mask': state['valid_actions_mask'].to(device)
                }
                
                # Get action from policy network
                with torch.no_grad():
                    action_probs = policy(
                        state_device['hand_matrix'],
                        state_device['discard_history']
                    )
                    
                    # Apply mask to action probabilities
                    action_probs = action_probs.squeeze()
                    action_probs = action_probs * state_device['valid_actions_mask']
                    
                    # Normalize probabilities
                    action_probs = action_probs / (action_probs.sum() + 1e-8)
                    
                    # Extract valid actions from the mask
                    valid_actions = torch.nonzero(state_device['valid_actions_mask']).squeeze().tolist()
                    if isinstance(valid_actions, int):
                        valid_actions = [valid_actions]  # Convert to list if only one valid action
                    
                    # Prioritize GIN and KNOCK actions if they're valid
                    if GIN in valid_actions:  # If GIN is valid, always take it
                        action = GIN
                        print(f"  Game {game+1}: Agent chose GIN action")
                        gin_taken += 1
                    elif KNOCK in valid_actions:  # If KNOCK is valid, always take it in evaluation
                        action = KNOCK
                        print(f"  Game {game+1}: Agent chose KNOCK action")
                        knock_taken += 1
                    else:
                        action = action_probs.argmax().item()
            # Random opponent's turn
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
                
            # Take action
            try:
                next_state, reward, done, truncated, info = env.step(action)
            except ValueError:
                # Handle the case where the environment returns 4 values instead of 5
                next_state, reward, done, info = env.step(action)
                truncated = False
            
            # Track rewards for player 0 (our agent)
            if env.current_player == 1:  # Just took action as player 0
                game_original_reward += info.get('original_reward', reward)
                game_shaped_reward += reward
            
            # Check if we won
            if done:
                if info.get('outcome') == 'win' and info.get('winner') == 0:
                    wins += 1
                    print(f"  Game {game+1}: Agent won the game!")
                else:
                    print(f"  Game {game+1}: Opponent won the game")
            
            # Update state
            state = next_state
        
        print(f"  Game {game+1} RESULT: {'WIN' if game+1 in [w+1 for w in range(wins)] else 'LOSS'}, Shaped reward: {game_shaped_reward:.2f}")
        total_shaped_reward += game_shaped_reward
        total_original_reward += game_original_reward
    
    # Print more detailed evaluation metrics
    print(f"\nEvaluation Summary - Wins: {wins}/{num_games}, Win Rate: {wins/num_games:.2f}")
    print(f"Avg Shaped Reward: {total_shaped_reward/num_games:.2f}, Avg Original Reward: {total_original_reward/num_games:.2f}")
    
    # Print action statistics
    if gin_opportunities > 0:
        print(f"GIN opportunities: {gin_opportunities}, taken: {gin_taken} ({gin_taken/gin_opportunities*100:.1f}%)")
    if knock_opportunities > 0:
        print(f"KNOCK opportunities: {knock_opportunities}, taken: {knock_taken} ({knock_taken/knock_opportunities*100:.1f}%)")
    print("===== END OF EVALUATION =====")
    
    return wins/num_games, total_shaped_reward/num_games, total_original_reward/num_games

def quick_train_mcts(env, device, policy_value_net, optimizer, scheduler=None, num_episodes=1000):
    """Train a policy-value network using MCTS"""
    win_rates = []
    
    # Create MCTS agent
    mcts_agent = MCTSAgent(policy_value_net, device, num_simulations=50)
    
    for episode in range(1, MCTS_EPISODES + 1):
        state = env.reset()
        done = False
        episode_reward = 0
        
        # Store states, actions, and rewards for training
        states = []
        actions = []
        rewards = []
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                # Get valid actions
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]  # Convert to list if only one valid action
                
                # Use MCTS to select action
                action = mcts_agent.select_action(state, valid_actions, training=True)
                
                # Store state and action
                states.append(state)
                actions.append(action)
            # Random opponent's turn
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Apply additional reward shaping
            if env.current_player == 1:  # Just took action as player 0
                episode_reward += reward
                rewards.append(reward)
            
            # Update state
            state = next_state
        
        # Train policy-value network
        if len(states) > 0:
            # Prepare batch
            batch_states = []
            batch_actions = torch.tensor(actions, device=device)
            batch_rewards = torch.tensor(rewards, device=device)
            
            for s in states:
                batch_states.append({
                    'hand_matrix': s['hand_matrix'].to(device),
                    'discard_history': s['discard_history'].to(device),
                    'valid_actions_mask': s['valid_actions_mask'].to(device)
                })
            
            # Compute policy loss
            policy_loss = 0
            value_loss = 0
            
            for i, s in enumerate(batch_states):
                policy_probs, value = policy_value_net(s['hand_matrix'], s['discard_history'])
                
                # Apply mask to policy probabilities
                policy_probs = policy_probs.squeeze()
                policy_probs = policy_probs * s['valid_actions_mask']
                
                # Normalize probabilities
                policy_probs = policy_probs / (policy_probs.sum() + 1e-8)
                
                # Compute policy loss (negative log likelihood)
                action_prob = policy_probs[batch_actions[i]]
                policy_loss += -torch.log(action_prob + 1e-8)
                
                # Compute value loss (mean squared error)
                value_loss += (value - batch_rewards[i]) ** 2
            
            # Compute total loss
            loss = policy_loss.mean() + 0.5 * value_loss.mean()
            
            # Update policy-value network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
        
        # Evaluate agent periodically
        if episode % EVAL_INTERVAL == 0:
            win_rate, avg_reward = evaluate_mcts(mcts_agent, env, num_games=10)
            win_rates.append(win_rate)
            print(f"\nEpisode {episode}/{MCTS_EPISODES} - Win rate: {win_rate:.2f}, Avg reward: {avg_reward:.2f}")
    
    return policy_value_net

def evaluate_mcts(policy_value_net, env, num_games=10):
    """Evaluate MCTS agent against random opponent"""
    wins = 0
    total_reward = 0
    
    # Action statistics
    gin_opportunities = 0
    knock_opportunities = 0
    gin_taken = 0
    knock_taken = 0
    
    print("\n===== DETAILED MCTS EVALUATION RESULTS =====")
    
    for game_num in range(num_games):
        print(f"\nEvaluating game {game_num+1}/{num_games}...")
        
        state = env.reset()
        done = False
        game_reward = 0
        game_win = False
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                # Get valid actions
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]  # Convert to list if only one valid action
                
                # Track GIN and KNOCK opportunities
                if GIN in valid_actions:
                    gin_opportunities += 1
                    print(f"  Game {game_num+1}: GIN opportunity detected!")
                if KNOCK in valid_actions:
                    knock_opportunities += 1
                    print(f"  Game {game_num+1}: KNOCK opportunity detected!")
                
                # Prioritize GIN and KNOCK actions if they're valid
                if GIN in valid_actions:  # If GIN is valid, always take it
                    action = GIN
                    gin_taken += 1
                    print(f"  Game {game_num+1}: Agent chose GIN action")
                elif KNOCK in valid_actions:  # If KNOCK is valid, always take it in evaluation
                    action = KNOCK
                    knock_taken += 1
                    print(f"  Game {game_num+1}: Agent chose KNOCK action")
                else:
                    # Use MCTS to select action
                    action = mcts_agent.select_action(state, valid_actions)
            # Random opponent's turn
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # CRITICAL FIX: Check if player 0 (our agent) won
            if done:
                if env.current_player == 1:  # Just took action as player 0
                    if 'outcome' in info and info['outcome'] is not None:
                        if info['outcome'] == 'win' or info['outcome'] == 'gin':
                            print(f"  Game {game_num+1}: Agent WON with {info['outcome']}! Player deadwood: {info['player_deadwood']}, Opponent deadwood: {info['opponent_deadwood']}")
                            wins += 1
                            game_win = True
                    elif reward > 0:  # Fallback if outcome not in info
                        print(f"  Game {game_num+1}: Agent WON with positive reward {reward}!")
                        wins += 1
                        game_win = True
                else:  # Opponent's action ended the game
                    print(f"  Game {game_num+1}: Opponent won the game")
            
            # Track rewards for player 0 (our agent)
            if env.current_player == 1:  # Just took action as player 0
                game_reward += reward
            
            # Update state
            state = next_state
        
        # Print game result
        if game_win:
            print(f"  Game {game_num+1} RESULT: WIN, Reward: {game_reward:.2f}")
        else:
            print(f"  Game {game_num+1} RESULT: LOSS, Reward: {game_reward:.2f}")
        
        total_reward += game_reward
    
    # Print more detailed evaluation metrics
    print(f"\nEvaluation Summary - Wins: {wins}/{num_games}, Win Rate: {wins/num_games:.2f}")
    print(f"Avg Reward: {total_reward/num_games:.2f}")
    
    # Print action statistics
    if gin_opportunities > 0:
        print(f"GIN opportunities: {gin_opportunities}, taken: {gin_taken} ({gin_taken/gin_opportunities*100:.1f}%)")
    if knock_opportunities > 0:
        print(f"KNOCK opportunities: {knock_opportunities}, taken: {knock_taken} ({knock_taken/knock_opportunities*100:.1f}%)")
    
    print("===== END OF EVALUATION =====\n")
    
    return wins / num_games, total_reward / num_games

class MCTSAgent:
    """Monte Carlo Tree Search agent for Gin Rummy"""
    def __init__(self, policy_value_net, device, num_simulations=50, c_puct=1.0):
        self.policy_value_net = policy_value_net
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct  # Exploration constant
        self.Qsa = {}  # Q values for state-action pairs: Q(s,a)
        self.Nsa = {}  # Visit count for state-action pairs: N(s,a)
        self.Ns = {}   # Visit count for states: N(s)
        self.Ps = {}   # Policy for states: P(s,a)
        
    def reset(self):
        """Reset the MCTS tree"""
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        
    def _state_to_key(self, state):
        """Convert state to hashable key"""
        hand_matrix = state['hand_matrix'].cpu().numpy().tobytes()
        discard_history = state['discard_history'].cpu().numpy().tobytes()
        return hash(hand_matrix + discard_history)
        
    def _get_action_probs(self, state, valid_actions, temp=1.0):
        """Get action probabilities from MCTS search"""
        state_key = self._state_to_key(state)
        
        # Run simulations
        for _ in range(self.num_simulations):
            self._search(state, valid_actions)
            
        # Calculate action probabilities
        action_visits = [(a, self.Nsa.get((state_key, a), 0)) for a in valid_actions]
        total_visits = sum(n for _, n in action_visits)
        
        # Convert visits to probabilities
        action_probs = torch.zeros(52 * 5, device=self.device)
        
        # If no visits, add a small probability to all valid actions
        if total_visits == 0:
            for a in valid_actions:
                action_probs[a] = 1.0 / len(valid_actions)
        else:
            for a, n in action_visits:
                action_probs[a] = n / total_visits
                
        return action_probs
        
    def _search(self, state, valid_actions):
        """Run a single MCTS simulation"""
        state_key = self._state_to_key(state)
        
        # Check if state is terminal
        if len(valid_actions) == 0:
            return 0
            
        # If state not in tree, add it
        if state_key not in self.Ps:
            # Move state to device
            state_device = {
                'hand_matrix': state['hand_matrix'].to(self.device),
                'discard_history': state['discard_history'].to(self.device),
                'valid_actions_mask': state['valid_actions_mask'].to(self.device)
            }
            
            # Get policy and value from network
            with torch.no_grad():
                policy_probs, value = self.policy_value_net(
                    state_device['hand_matrix'],
                    state_device['discard_history']
                )
                
                # Apply mask to policy probabilities
                policy_probs = policy_probs.squeeze()
                policy_probs = policy_probs * state_device['valid_actions_mask']
                
                # Normalize probabilities
                policy_probs = policy_probs / (policy_probs.sum() + 1e-8)
                
            # Store policy and initialize counts
            self.Ps[state_key] = policy_probs.cpu().numpy()
            self.Ns[state_key] = 0
            
            # Return value from network's evaluation
            return value.item()
            
        # Select action with highest UCB score
        best_score = -float('inf')
        best_action = None
        
        # Prioritize GIN and KNOCK actions if they're valid, but still update visit counts
        if GIN in valid_actions:
            best_action = GIN
        elif KNOCK in valid_actions:
            best_action = KNOCK
        else:
            # Otherwise, use UCB to select action
            for a in valid_actions:
                # Skip invalid actions
                if a not in valid_actions:
                    continue
                    
                # Calculate UCB score
                if (state_key, a) in self.Qsa:
                    q = self.Qsa[(state_key, a)]
                    u = self.c_puct * self.Ps[state_key][a] * math.sqrt(self.Ns[state_key]) / (1 + self.Nsa.get((state_key, a), 0))
                    score = q + u
                else:
                    # Prioritize unexplored actions
                    score = self.c_puct * self.Ps[state_key][a] * math.sqrt(self.Ns[state_key] + 1e-8)
                    
                # Update best action
                if score > best_score:
                    best_score = score
                    best_action = a
                    
        # If no valid actions, return 0
        if best_action is None:
            return 0
            
        return self._evaluate_and_backup(state, best_action, valid_actions)
        
    def _evaluate_and_backup(self, state, action, valid_actions):
        """Evaluate action and backup values"""
        state_key = self._state_to_key(state)
        
        # Create a temporary environment for simulation
        temp_env = ImprovedGinRummyEnv(reward_shaping=True)
        
        # Set the environment state to match the current state
        temp_env.reset()
        temp_env.current_player = 0  # Ensure it's our agent's turn
        
        # Extract player's hand from hand matrix
        player_hand = []
        for suit in range(4):
            for rank in range(13):
                if state['hand_matrix'][0, suit, rank] > 0:
                    card = suit * 13 + rank
                    player_hand.append(card)
        
        # Set player's hand
        temp_env.player_hands[0] = player_hand
        
        # For opponent's hand, we can use the opponent model if available
        if 'opponent_model' in state:
            opponent_hand = []
            for card in range(52):
                if state['opponent_model'][card] > 0:
                    opponent_hand.append(card)
            temp_env.player_hands[1] = opponent_hand
        else:
            # If opponent model is not available, create a random hand
            remaining_cards = [i for i in range(52) if i not in player_hand]
            temp_env.player_hands[1] = random.sample(remaining_cards, 10)
        
        # Set discard pile
        discard_pile = []
        for i in range(10):  # Last 10 discards
            for card in range(52):
                if state['discard_history'][0, i, card] > 0:
                    discard_pile.append(card)
        temp_env.discard_pile = discard_pile
        
        # Set deck
        temp_env.deck = [i for i in range(52) if i not in temp_env.player_hands[0] and i not in temp_env.player_hands[1] and i not in temp_env.discard_pile]
        
        # Take action in temporary environment
        next_state, reward, done, _, info = temp_env.step(action)
        
        # Get value of next state
        if done:
            value = reward
        else:
            # Get valid actions for next state
            next_valid_actions = torch.nonzero(next_state['valid_actions_mask']).squeeze().tolist()
            if isinstance(next_valid_actions, int):
                next_valid_actions = [next_valid_actions]
                
            # Recursively search from next state
            value = -self._search(next_state, next_valid_actions)  # Negative because it's opponent's turn
            
        # Update statistics
        if (state_key, action) in self.Qsa:
            self.Qsa[(state_key, action)] = (self.Nsa.get((state_key, action), 0) * self.Qsa[(state_key, action)] + value) / (self.Nsa.get((state_key, action), 0) + 1)
            self.Nsa[(state_key, action)] = self.Nsa.get((state_key, action), 0) + 1
            
        self.Ns[state_key] = self.Ns.get(state_key, 0) + 1
        
        return value
        
    def select_action(self, state, valid_actions, training=False):
        """Select action using MCTS"""
        # Prioritize GIN and KNOCK actions if they're valid
        if GIN in valid_actions:
            return GIN
        elif KNOCK in valid_actions:
            return KNOCK
            
        # Get action probabilities from MCTS
        action_probs = self._get_action_probs(state, valid_actions)
        
        # In training, add some exploration
        if training:
            # Add Dirichlet noise to action probabilities
            noise = torch.zeros_like(action_probs)
            for a in valid_actions:
                noise[a] = 0.03  # Small noise for exploration
            action_probs = 0.75 * action_probs + 0.25 * noise
            action_probs = action_probs / action_probs.sum()
            
            # Sample action from probabilities
            action_idx = torch.multinomial(action_probs, 1).item()
            if action_idx in valid_actions:
                return action_idx
                
        # In evaluation, choose action with highest probability
        return action_probs.argmax().item()

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm"""
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
        # Input dimensions
        self.hand_channels = 4  # 4 suits
        self.hand_ranks = 13  # 13 ranks
        self.discard_seq_len = 10  # Last 10 discards
        self.discard_features = 52  # One-hot encoding of cards
        
        # Convolutional layers for hand processing
        self.hand_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # LSTM for discard history processing
        self.discard_lstm = nn.LSTM(
            input_size=self.discard_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # Calculate the flattened size of the conv output
        # For a 4x13 input with the given conv layers, the output will be 64 x 4 x 13
        conv_output_size = 64 * 4 * 13
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, 110)  # Match the environment's action space (110 actions)
        
    def forward(self, hand_matrix, discard_history):
        # Process hand with CNN
        batch_size = hand_matrix.size(0)
        
        # Ensure hand_matrix has the right shape [batch_size, channels, height, width]
        if len(hand_matrix.shape) == 3:  # [batch_size, height, width]
            hand_matrix = hand_matrix.unsqueeze(1)  # Add channel dimension
            
        hand_features = self.hand_conv(hand_matrix)
        
        # Process discard history with LSTM
        # Ensure discard_history has the right shape [batch_size, seq_len, features]
        if len(discard_history.shape) == 2:
            discard_history = discard_history.unsqueeze(0)  # Add batch dimension if missing
            
        discard_out, _ = self.discard_lstm(discard_history)
        discard_features = discard_out[:, -1, :]  # Take the last output
        
        # Ensure both tensors have the same batch dimension
        if hand_features.size(0) != discard_features.size(0):
            if hand_features.size(0) == 1 and discard_features.size(0) > 1:
                hand_features = hand_features.expand(discard_features.size(0), -1)
            elif discard_features.size(0) == 1 and hand_features.size(0) > 1:
                discard_features = discard_features.expand(hand_features.size(0), -1)
        
        # Concatenate features
        combined = torch.cat((hand_features, discard_features), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # Action probabilities
        action_probs = F.softmax(self.action_head(x), dim=1)
        
        return action_probs

class DQNetwork(nn.Module):
    """Deep Q-Network for Gin Rummy"""
    def __init__(self):
        super(DQNetwork, self).__init__()
        
        # Input dimensions
        self.hand_channels = 4  # 4 suits
        self.hand_ranks = 13  # 13 ranks
        self.discard_seq_len = 10  # Last 10 discards
        self.discard_features = 52  # One-hot encoding of cards
        
        # Convolutional layers for hand processing
        self.hand_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # LSTM for discard history processing
        self.discard_lstm = nn.LSTM(
            input_size=self.discard_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # Calculate the flattened size of the conv output
        # For a 4x13 input with the given conv layers, the output will be 64 x 4 x 13
        conv_output_size = 64 * 4 * 13
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q_head = nn.Linear(128, 110)  # Match the environment's action space (110 actions)
        
    def forward(self, hand_matrix, discard_history):
        # Process hand with CNN
        batch_size = hand_matrix.size(0)
        
        # Ensure hand_matrix has the right shape [batch_size, channels, height, width]
        if len(hand_matrix.shape) == 3:  # [batch_size, height, width]
            hand_matrix = hand_matrix.unsqueeze(1)  # Add channel dimension
            
        hand_features = self.hand_conv(hand_matrix)
        
        # Process discard history with LSTM
        # Ensure discard_history has the right shape [batch_size, seq_len, features]
        if len(discard_history.shape) == 2:
            discard_history = discard_history.unsqueeze(0)  # Add batch dimension if missing
            
        discard_out, _ = self.discard_lstm(discard_history)
        discard_features = discard_out[:, -1, :]  # Take the last output
        
        # Ensure both tensors have the same batch dimension
        if hand_features.size(0) != discard_features.size(0):
            if hand_features.size(0) == 1 and discard_features.size(0) > 1:
                hand_features = hand_features.expand(discard_features.size(0), -1)
            elif discard_features.size(0) == 1 and hand_features.size(0) > 1:
                discard_features = discard_features.expand(hand_features.size(0), -1)
        
        # Concatenate features
        combined = torch.cat((hand_features, discard_features), dim=1)
        
        # Fully connected layers
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        
        # Q-values
        q_values = self.q_head(x)
        
        return q_values

class PolicyValueNetwork(nn.Module):
    """Policy-Value Network for MCTS algorithm"""
    def __init__(self):
        super(PolicyValueNetwork, self).__init__()
        
        # Input dimensions
        self.hand_channels = 4  # 4 suits
        self.hand_ranks = 13  # 13 ranks
        self.discard_seq_len = 10  # Last 10 discards
        self.discard_features = 52  # One-hot encoding of cards
        
        # Convolutional layers for hand processing
        self.hand_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # LSTM for discard history processing
        self.discard_lstm = nn.LSTM(
            input_size=self.discard_features,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        # Calculate the flattened size of the conv output
        # For a 4x13 input with the given conv layers, the output will be 64 x 4 x 13
        conv_output_size = 64 * 4 * 13
        
        # Shared representation
        self.shared_fc = nn.Sequential(
            nn.Linear(conv_output_size + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Policy head
        self.policy_head = nn.Linear(128, 110)  # Match the environment's action space (110 actions)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()  # Output between -1 and 1
        )
        
    def forward(self, hand_matrix, discard_history):
        # Process hand with CNN
        batch_size = hand_matrix.size(0)
        
        # Ensure hand_matrix has the right shape [batch_size, channels, height, width]
        if len(hand_matrix.shape) == 3:  # [batch_size, height, width]
            hand_matrix = hand_matrix.unsqueeze(1)  # Add channel dimension
            
        hand_features = self.hand_conv(hand_matrix)
        
        # Process discard history with LSTM
        # Ensure discard_history has the right shape [batch_size, seq_len, features]
        if len(discard_history.shape) == 2:
            discard_history = discard_history.unsqueeze(0)  # Add batch dimension if missing
            
        discard_out, _ = self.discard_lstm(discard_history)
        discard_features = discard_out[:, -1, :]  # Take the last output
        
        # Ensure both tensors have the same batch dimension
        if hand_features.size(0) != discard_features.size(0):
            if hand_features.size(0) == 1 and discard_features.size(0) > 1:
                hand_features = hand_features.expand(discard_features.size(0), -1)
            elif discard_features.size(0) == 1 and hand_features.size(0) > 1:
                discard_features = discard_features.expand(hand_features.size(0), -1)
        
        # Concatenate features
        combined = torch.cat((hand_features, discard_features), dim=1)
        
        # Shared representation
        shared_features = self.shared_fc(combined)
        
        # Policy output
        policy_logits = self.policy_head(shared_features)
        policy_probs = F.softmax(policy_logits, dim=1)
        
        # Value output
        value = self.value_head(shared_features)
        
        return policy_probs, value

def evaluate_agent(model, env, device, num_games=20, num_simulations=50, verbose=False):
    """
    Generic evaluation function that detects the model type and calls the appropriate evaluation function.
    
    Args:
        model: The model to evaluate (DQNetwork, PolicyNetwork, or PolicyValueNetwork)
        env: The environment to evaluate in
        device: The device to run on
        num_games: Number of games to evaluate
        num_simulations: Number of simulations for MCTS (only used if model is PolicyValueNetwork)
        verbose: Whether to print detailed information
        
    Returns:
        win_rate: The win rate of the agent
        avg_reward: The average reward per game
    """
    if isinstance(model, DQNetwork):
        return evaluate_dqn(model, device, env, num_games)
    elif isinstance(model, PolicyNetwork):
        # For REINFORCE
        wins = 0
        total_reward = 0
        
        model.eval()
        for game in range(num_games):
            state = env.reset()
            done = False
            
            while not done:
                # Get valid actions
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                
                # If it's the agent's turn
                if env.current_player == 0:
                    # Prioritize GIN and KNOCK actions
                    if GIN in valid_actions:
                        action = GIN
                    elif KNOCK in valid_actions:
                        action = KNOCK
                    else:
                        # Move state to device
                        state_device = {
                            'hand_matrix': state['hand_matrix'].to(device),
                            'discard_history': state['discard_history'].to(device),
                            'valid_actions_mask': state['valid_actions_mask'].to(device)
                        }
                        
                        # Get action probabilities from policy network
                        with torch.no_grad():
                            action_probs = model(
                                state_device['hand_matrix'],
                                state_device['discard_history']
                            ).squeeze()
                        
                        # Apply mask to action probabilities
                        masked_probs = action_probs * state_device['valid_actions_mask']
                        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
                        
                        # Choose best action
                        action = masked_probs.argmax().item()
                else:
                    # Opponent's turn - random action
                    action = random.choice(valid_actions)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                state = next_state
                
                # Track reward
                if env.current_player == 0:  # Only count reward for our agent
                    total_reward += reward
            
            # Check if agent won
            if info['winner'] == 0:
                wins += 1
                
            if verbose:
                print(f"Game {game+1}: {'Win' if info['winner'] == 0 else 'Loss'}")
        
        win_rate = wins / num_games
        avg_reward = total_reward / num_games
        
        if verbose:
            print(f"Win Rate: {win_rate:.2f}")
            print(f"Average Reward: {avg_reward:.2f}")
        
        return win_rate, avg_reward
    
    elif isinstance(model, PolicyValueNetwork):
        # For MCTS
        agent = MCTSAgent(model, device, num_simulations=num_simulations)
        wins = 0
        total_reward = 0
        
        for game in range(num_games):
            state = env.reset()
            done = False
            agent.reset()  # Reset MCTS tree
            
            while not done:
                # Get valid actions
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                
                # If it's the agent's turn
                if env.current_player == 0:
                    # Move state to device
                    state_device = {
                        'hand_matrix': state['hand_matrix'].to(device),
                        'discard_history': state['discard_history'].to(device),
                        'valid_actions_mask': state['valid_actions_mask'].to(device)
                    }
                    
                    # Get action from MCTS agent
                    action = agent.select_action(state_device, valid_actions, training=False)
                else:
                    # Opponent's turn - random action
                    action = random.choice(valid_actions)
                
                # Take action
                next_state, reward, done, info = env.step(action)
                state = next_state
                
                # Track reward
                if env.current_player == 0:  # Only count reward for our agent
                    total_reward += reward
            
            # Check if agent won
            if info['winner'] == 0:
                wins += 1
                
            if verbose:
                print(f"Game {game+1}: {'Win' if info['winner'] == 0 else 'Loss'}")
        
        win_rate = wins / num_games
        avg_reward = total_reward / num_games
        
        if verbose:
            print(f"Win Rate: {win_rate:.2f}")
            print(f"Average Reward: {avg_reward:.2f}")
        
        return win_rate, avg_reward
    
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

def main():
    """Main function to run the training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate Gin Rummy agents')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'reinforce', 'mcts'],
                        help='Agent type to train/evaluate')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate the agent, no training')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to model file for evaluation')
    parser.add_argument('--num-games', type=int, default=20,
                        help='Number of games to evaluate')
    parser.add_argument('--reward-shaping', action='store_true',
                        help='Use reward shaping during training')
    parser.add_argument('--deadwood-scale', type=float, default=0.02,
                        help='Scale factor for deadwood reduction rewards')
    parser.add_argument('--win-reward', type=float, default=1.5,
                        help='Reward for winning a game')
    parser.add_argument('--gin-reward', type=float, default=2.0,
                        help='Reward for winning with gin')
    parser.add_argument('--knock-reward', type=float, default=0.75,
                        help='Reward for knocking')
    
    args = parser.parse_args()
    
    # Set up environment with configurable rewards
    env = ImprovedGinRummyEnv(
        reward_shaping=args.reward_shaping,
        deadwood_reward_scale=args.deadwood_scale,
        win_reward=args.win_reward,
        gin_reward=args.gin_reward,
        knock_reward=args.knock_reward
    )
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    print(f"Using device: {device}")
    
    if args.eval_only:
        # Evaluation mode
        if args.agent == 'dqn':
            model = DQNetwork().to(device)
            model_path = args.model_path or "models/improved_dqn_final.pt"
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded DQN model from {model_path}")
                win_rate, avg_reward = evaluate_agent(model, env, device, args.num_games)
            except Exception as e:
                print(f"Error loading model: {e}")
        
        elif args.agent == 'reinforce':
            model = PolicyNetwork().to(device)
            model_path = args.model_path or "models/improved_reinforce_final.pt"
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded REINFORCE model from {model_path}")
                win_rate, avg_reward = evaluate_agent(model, env, device, args.num_games)
            except Exception as e:
                print(f"Error loading model: {e}")
        
        elif args.agent == 'mcts':
            model = PolicyValueNetwork().to(device)
            model_path = args.model_path or "models/improved_mcts_final.pt"
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded MCTS model from {model_path}")
                win_rate, avg_reward = evaluate_agent(model, env, device, args.num_games)
            except Exception as e:
                print(f"Error loading model: {e}")
    
    else:
        # Training mode
        if args.agent == 'dqn':
            model = quick_train_dqn(env, device)
            win_rate, avg_reward = evaluate_agent(model, env, device, args.num_games)
        
        elif args.agent == 'reinforce':
            model = quick_train_reinforce(env, device)
            win_rate, avg_reward = evaluate_agent(model, env, device, args.num_games)
        
        elif args.agent == 'mcts':
            model = PolicyValueNetwork().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
            model = quick_train_mcts(env, device, model, optimizer, scheduler)
            win_rate, avg_reward = evaluate_agent(model, env, device, args.num_games)
    
    print(f"Final results for {args.agent.upper()}:")
    print(f"  Win rate: {win_rate:.2f}")
    print(f"  Average reward: {avg_reward:.2f}")

if __name__ == "__main__":
    main() 