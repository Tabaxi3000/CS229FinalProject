#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN

class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE."""
    
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        
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
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
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
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Apply softmax to get action probabilities
        return F.softmax(x, dim=1)

def train_reinforce(num_episodes=5000, lr=0.0001, gamma=0.99, entropy_beta=0.01,
                   reward_shaping=True, deadwood_reward_scale=0.01, win_reward=1.0,
                   gin_reward=1.5, knock_reward=0.5, eval_interval=100,
                   save_interval=1000, model_path="models/improved_reinforce.pt"):
    """Train a REINFORCE agent for Gin Rummy."""
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
    
    # Set up policy network and optimizer
    policy_net = PolicyNetwork().to(device)
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    
    # Training metrics
    episode_rewards = []
    losses = []
    win_rates = []
    
    # Create directories for saving models and metrics
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training REINFORCE")
    
    for episode in pbar:
        # Reset environment
        state = env.reset()
        
        # Storage for episode data
        log_probs = []
        rewards = []
        
        # Play one episode
        done = False
        episode_reward = 0
        
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
                
                # Get action probabilities from policy network
                action_probs = policy_net(
                    state_device['hand_matrix'].unsqueeze(0),
                    state_device['discard_history'].unsqueeze(0)
                ).squeeze()
                
                # Apply mask to ensure we only sample from valid actions
                masked_probs = action_probs * state_device['valid_actions_mask']
                masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
                
                # Sample action from policy
                m = torch.distributions.Categorical(masked_probs)
                action = m.sample().item()
                
                # Store log probability of the action taken
                log_prob = m.log_prob(torch.tensor(action, device=device))
                log_probs.append(log_prob)
            else:
                # Opponent's turn - random action
                action = np.random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Store reward if it's the agent's turn
            if env.current_player == 0:
                rewards.append(reward)
                episode_reward += reward
            
            # Update state
            state = next_state
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        # Convert to tensor
        returns = torch.tensor(returns, dtype=torch.float, device=device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.stack(policy_loss).sum()
        
        # Add entropy bonus to encourage exploration
        if entropy_beta > 0:
            # Get action probabilities for all states visited
            entropy_loss = 0
            for i in range(len(log_probs)):
                action_probs = policy_net(
                    state_device['hand_matrix'].unsqueeze(0),
                    state_device['discard_history'].unsqueeze(0)
                ).squeeze()
                
                # Apply mask
                masked_probs = action_probs * state_device['valid_actions_mask']
                masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
                
                # Calculate entropy
                entropy = -(masked_probs * torch.log(masked_probs + 1e-8)).sum()
                entropy_loss -= entropy
            
            # Add entropy loss to policy loss
            policy_loss += entropy_beta * entropy_loss
        
        # Optimize
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Track metrics
        episode_rewards.append(episode_reward)
        losses.append(policy_loss.item())
        
        # Evaluate agent
        if (episode + 1) % eval_interval == 0:
            win_rate = evaluate_reinforce(policy_net, env, device, num_games=20)[0]
            win_rates.append(win_rate)
            
            # Update progress bar
            pbar.set_postfix({
                'win_rate': f'{win_rate:.2f}',
                'reward': f'{episode_reward:.2f}',
                'loss': f'{policy_loss.item():.4f}'
            })
            
            # Save metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'losses': losses,
                'win_rates': win_rates,
                'eval_episodes': list(range(eval_interval, episode + 2, eval_interval))
            }
            
            with open('metrics/reinforce_training_metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            # Plot metrics
            plot_metrics(episode_rewards, losses, win_rates, eval_interval)
        
        # Save model
        if (episode + 1) % save_interval == 0 and args.save:
            torch.save(policy_net.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    if args.save:
        torch.save(policy_net.state_dict(), model_path)
        print(f"Final model saved to {model_path}")
    
    return policy_net, episode_rewards, losses, win_rates

def evaluate_reinforce(policy_net, env, device, num_games=20, verbose=False):
    """Evaluate a REINFORCE agent."""
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
                    # Get action probabilities
                    with torch.no_grad():
                        action_probs = policy_net(
                            state_device['hand_matrix'].unsqueeze(0),
                            state_device['discard_history'].unsqueeze(0)
                        ).squeeze()
                        
                        # Apply mask
                        masked_probs = action_probs * state_device['valid_actions_mask']
                        masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
                        
                        # Choose best action
                        action = masked_probs.argmax().item()
            else:
                # Opponent's turn - random action
                action = np.random.choice(valid_actions)
            
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

def plot_metrics(rewards, losses, win_rates, eval_interval):
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
    
    # Empty subplot for consistency
    plt.subplot(2, 2, 4)
    plt.title('Reserved for Future Metrics')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("plots/reinforce_training_metrics.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a REINFORCE agent for Gin Rummy')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--entropy-beta', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--reward-shaping', action='store_true', help='Use reward shaping')
    parser.add_argument('--deadwood-reward-scale', type=float, default=0.01, help='Scale for deadwood reduction reward')
    parser.add_argument('--win-reward', type=float, default=1.0, help='Reward for winning')
    parser.add_argument('--gin-reward', type=float, default=1.5, help='Additional reward for gin')
    parser.add_argument('--knock-reward', type=float, default=0.5, help='Additional reward for knock')
    parser.add_argument('--eval-interval', type=int, default=100, help='Interval for evaluation')
    parser.add_argument('--save-interval', type=int, default=1000, help='Interval for saving model')
    parser.add_argument('--save', action='store_true', help='Save model')
    parser.add_argument('--model-path', type=str, default='models/improved_reinforce.pt', help='Path to save model')
    
    args = parser.parse_args()
    
    train_reinforce(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        entropy_beta=args.entropy_beta,
        reward_shaping=args.reward_shaping,
        deadwood_reward_scale=args.deadwood_reward_scale,
        win_reward=args.win_reward,
        gin_reward=args.gin_reward,
        knock_reward=args.knock_reward,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        model_path=args.model_path
    ) 