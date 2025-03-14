#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import os
import json
import matplotlib.pyplot as plt
from collections import deque
from tqdm import tqdm
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN

# Constants
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 10000
LEARNING_RATE = 0.0001

class ReplayMemory:
    """Experience replay memory for DQN."""
    
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

class DQNetwork(nn.Module):
    """Deep Q-Network for Gin Rummy."""
    
    def __init__(self):
        super(DQNetwork, self).__init__()
        
        # Convolutional layers for processing hand matrix
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # LSTM for processing discard history
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 13 + 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 110)  # 110 actions
    
    def forward(self, hand_matrix, discard_history):
        """Forward pass through the network."""
        # Process hand matrix through convolutional layers
        x = hand_matrix.float()
        
        # Ensure hand_matrix has the right shape [batch, channel, height, width]
        if x.dim() == 3:  # If [batch, 4, 13]
            x = x.unsqueeze(1)  # Add channel dim [batch, 1, 4, 13]
        
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        
        # Flatten the output
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Process discard history through LSTM
        discard_history = discard_history.float()
        lstm_out, _ = self.lstm(discard_history)
        
        # Take the last output from the LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Concatenate features
        combined = torch.cat([x, lstm_out], dim=1)
        
        # Pass through fully connected layers
        x = torch.relu(self.fc1(combined))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

def train_dqn(num_episodes=5000, batch_size=BATCH_SIZE, gamma=GAMMA, 
             epsilon_start=EPSILON_START, epsilon_end=EPSILON_END, 
             epsilon_decay=EPSILON_DECAY, target_update=TARGET_UPDATE, 
             memory_size=MEMORY_SIZE, learning_rate=LEARNING_RATE,
             reward_shaping=True, deadwood_reward_scale=0.01, win_reward=1.0,
             gin_reward=1.5, knock_reward=0.5, eval_interval=100,
             save_interval=1000, model_path="models/improved_dqn.pt"):
    """Train a DQN agent for Gin Rummy."""
    # Set up environment
    env = ImprovedGinRummyEnv(reward_shaping=reward_shaping, 
                             deadwood_reward_scale=deadwood_reward_scale,
                             win_reward=win_reward,
                             gin_reward=gin_reward,
                             knock_reward=knock_reward)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    print(f"Using device: {device}")
    
    # Set up networks
    policy_net = DQNetwork().to(device)
    target_net = DQNetwork().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    # Set up optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    
    # Set up replay memory
    memory = ReplayMemory(memory_size)
    
    # Training metrics
    epsilon = epsilon_start
    episode_rewards = []
    losses = []
    win_rates = []
    epsilons = []
    
    # Create directories for saving models and metrics
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training DQN")
    
    for episode in pbar:
        # Reset environment
        state = env.reset()
        
        # Play one episode
        done = False
        episode_reward = 0
        episode_loss = 0
        steps = 0
        
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
                
                # Epsilon-greedy action selection
                if random.random() < epsilon:
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
            else:
                # Opponent's turn - random action
                action = random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Store transition in replay memory if it's the agent's turn
            if env.current_player == 0:
                memory.push(state, action, reward, next_state, done)
                episode_reward += reward
            
            # Update state
            state = next_state
            steps += 1
            
            # Train on a batch of transitions if enough samples
            if len(memory) >= batch_size and env.current_player == 0:
                # Sample a batch
                transitions = memory.sample(batch_size)
                
                # Prepare batch
                batch_state = []
                batch_action = []
                batch_reward = []
                batch_next_state = []
                batch_done = []
                
                for s, a, r, ns, d in transitions:
                    batch_state.append(s)
                    batch_action.append(a)
                    batch_reward.append(r)
                    batch_next_state.append(ns)
                    batch_done.append(d)
                
                # Convert to tensors
                batch_action = torch.tensor(batch_action, dtype=torch.long, device=device).unsqueeze(1)
                batch_reward = torch.tensor(batch_reward, dtype=torch.float, device=device)
                batch_done = torch.tensor(batch_done, dtype=torch.float, device=device)
                
                # Prepare state and next_state batches
                batch_hand_matrix = torch.stack([s['hand_matrix'] for s in batch_state]).to(device)
                batch_discard_history = torch.stack([s['discard_history'] for s in batch_state]).to(device)
                
                batch_next_hand_matrix = torch.stack([ns['hand_matrix'] for ns in batch_next_state]).to(device)
                batch_next_discard_history = torch.stack([ns['discard_history'] for ns in batch_next_state]).to(device)
                batch_next_valid_actions_mask = torch.stack([ns['valid_actions_mask'] for ns in batch_next_state]).to(device)
                
                # Compute Q(s_t, a)
                q_values = policy_net(batch_hand_matrix, batch_discard_history).gather(1, batch_action)
                
                # Compute V(s_{t+1}) for all next states
                with torch.no_grad():
                    # Get Q-values for all actions in next states
                    next_q_values = target_net(batch_next_hand_matrix, batch_next_discard_history)
                    
                    # Apply mask to next Q-values
                    next_q_values[~batch_next_valid_actions_mask.bool()] = float('-inf')
                    
                    # Get maximum Q-value for each next state
                    next_q_values_max = next_q_values.max(1)[0].detach()
                    
                    # Compute the expected Q values
                    expected_q_values = batch_reward + (gamma * next_q_values_max * (1 - batch_done))
                
                # Compute loss
                criterion = nn.SmoothL1Loss()
                loss = criterion(q_values, expected_q_values.unsqueeze(1))
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                episode_loss += loss.item()
        
        # Update target network
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
        
        # Track metrics
        episode_rewards.append(episode_reward)
        losses.append(episode_loss / max(1, steps))
        epsilons.append(epsilon)
        
        # Evaluate agent
        if (episode + 1) % eval_interval == 0:
            win_rate = evaluate_dqn(policy_net, env, device, num_games=20)[0]
            win_rates.append(win_rate)
            
            # Update progress bar
            pbar.set_postfix({
                'win_rate': f'{win_rate:.2f}',
                'reward': f'{episode_reward:.2f}',
                'loss': f'{episode_loss/max(1, steps):.4f}',
                'epsilon': f'{epsilon:.4f}'
            })
            
            # Save metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'losses': losses,
                'win_rates': win_rates,
                'epsilons': epsilons,
                'eval_episodes': list(range(eval_interval, episode + 2, eval_interval))
            }
            
            with open('metrics/dqn_training_metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            # Plot metrics
            plot_metrics(episode_rewards, losses, win_rates, epsilons, eval_interval)
        
        # Save model
        if (episode + 1) % save_interval == 0 and args.save:
            torch.save(policy_net.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    if args.save:
        torch.save(policy_net.state_dict(), model_path)
        print(f"Final model saved to {model_path}")
    
    return policy_net, episode_rewards, losses, win_rates, epsilons

def evaluate_dqn(policy_net, env, device, num_games=20, verbose=False):
    """Evaluate a DQN agent."""
    policy_net.eval()
    
    wins = 0
    total_reward = 0
    
    for game in range(num_games):
        state = env.reset()
        done = False
        game_reward = 0
        
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
                
                # Prioritize GIN and KNOCK actions
                if GIN in valid_actions:
                    action = GIN
                elif KNOCK in valid_actions:
                    action = KNOCK
                else:
                    # Get Q-values
                    with torch.no_grad():
                        q_values = policy_net(
                            state_device['hand_matrix'],
                            state_device['discard_history']
                        )
                        
                        # Apply mask to Q-values
                        masked_q_values = q_values.squeeze().clone()
                        masked_q_values[~state_device['valid_actions_mask'].bool()] = float('-inf')
                        action = masked_q_values.argmax().item()
            else:
                # Opponent's turn - random action
                action = random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Track reward for agent's actions
            if env.current_player == 0:
                game_reward += reward
            
            # Update state
            state = next_state
        
        # Check if agent won
        if 'outcome' in info:
            if info['outcome'] == 'win' or info['outcome'] == 'gin':
                if env.current_player == 1:  # Agent's turn just ended
                    wins += 1
            else:
                if env.current_player == 0:  # Opponent's turn just ended
                    wins += 1
        
        total_reward += game_reward
        
        if verbose:
            print(f"Game {game+1}: {'Win' if (('outcome' in info and (info['outcome'] == 'win' or info['outcome'] == 'gin') and env.current_player == 1) or ('outcome' in info and info['outcome'] == 'loss' and env.current_player == 0)) else 'Loss'}, Reward: {game_reward:.4f}")
    
    policy_net.train()
    return wins / num_games, total_reward / num_games

def plot_metrics(rewards, losses, win_rates, epsilons, eval_interval):
    """Plot training metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot losses
    plt.subplot(2, 2, 2)
    plt.plot(losses)
    plt.title('Episode Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot win rates
    plt.subplot(2, 2, 3)
    plt.plot(range(eval_interval, len(win_rates) * eval_interval + 1, eval_interval), win_rates, marker='o')
    plt.title('Win Rates')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    # Plot epsilons
    plt.subplot(2, 2, 4)
    plt.plot(epsilons)
    plt.title('Epsilon Values')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig("plots/dqn_training_metrics.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a DQN agent for Gin Rummy')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--gamma', type=float, default=GAMMA, help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=EPSILON_START, help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=EPSILON_END, help='Minimum epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=EPSILON_DECAY, help='Epsilon decay rate')
    parser.add_argument('--target-update', type=int, default=TARGET_UPDATE, help='Interval for target network update')
    parser.add_argument('--memory-size', type=int, default=MEMORY_SIZE, help='Size of replay memory')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('--reward-shaping', action='store_true', help='Use reward shaping')
    parser.add_argument('--deadwood-reward-scale', type=float, default=0.01, help='Scale for deadwood reduction reward')
    parser.add_argument('--win-reward', type=float, default=1.0, help='Reward for winning')
    parser.add_argument('--gin-reward', type=float, default=1.5, help='Additional reward for gin')
    parser.add_argument('--knock-reward', type=float, default=0.5, help='Additional reward for knock')
    parser.add_argument('--eval-interval', type=int, default=100, help='Interval for evaluation')
    parser.add_argument('--save-interval', type=int, default=1000, help='Interval for saving model')
    parser.add_argument('--save', action='store_true', help='Save model')
    parser.add_argument('--model-path', type=str, default='models/improved_dqn.pt', help='Path to save model')
    
    args = parser.parse_args()
    
    train_dqn(
        num_episodes=args.episodes,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        memory_size=args.memory_size,
        learning_rate=args.lr,
        reward_shaping=args.reward_shaping,
        deadwood_reward_scale=args.deadwood_reward_scale,
        win_reward=args.win_reward,
        gin_reward=args.gin_reward,
        knock_reward=args.knock_reward,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        model_path=args.model_path
    ) 