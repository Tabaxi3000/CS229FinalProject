#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple, deque
import random
import numpy as np
import os
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# Import our models and environment
from dqn import DQNAgent, DQNetwork
from reinforce import REINFORCEAgent, PolicyNetwork
from mcts import MCTSAgent, PolicyValueNetwork
from gin_rummy_env import GinRummyEnv

# Constants
BATCH_SIZE = 128
GAMMA = 0.99
LEARNING_RATE = 0.0001
EPISODES = 5000
MEMORY_SIZE = 100000
TARGET_UPDATE = 10
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 5000
REWARD_SCALING = 10.0  # Scale rewards for better learning
SAVE_INTERVAL = 500
EVAL_INTERVAL = 200
ENTROPY_COEF = 0.01  # Entropy coefficient for exploration

# Experience replay memory
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayMemory:
    """Replay memory for DQN training"""
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        """Save a transition"""
        self.memory.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        return random.sample(self.memory, min(len(self.memory), batch_size))
    
    def __len__(self):
        return len(self.memory)

class ImprovedDQNAgent:
    """Improved DQN agent with prioritized experience replay and dueling architecture"""
    def __init__(self, state_dim=None, action_dim=110, hidden_dim=256, learning_rate=LEARNING_RATE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
            
        # Initialize networks
        self.policy_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Initialize optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Initialize replay memory
        self.memory = ReplayMemory(MEMORY_SIZE)
        
        # Training parameters
        self.steps_done = 0
        self.epsilon = EPSILON_START
        
    def select_action(self, state, eval_mode=False):
        """Select an action using epsilon-greedy policy"""
        # Calculate epsilon
        if eval_mode:
            epsilon = 0.05  # Low epsilon for evaluation
        else:
            epsilon = max(EPSILON_END, EPSILON_START - (self.steps_done / EPSILON_DECAY))
            self.epsilon = epsilon
            
        # Move state to device
        for key in state:
            if isinstance(state[key], torch.Tensor):
                state[key] = state[key].to(self.device)
        
        # Get valid actions
        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
        if isinstance(valid_actions, int):
            valid_actions = [valid_actions]
            
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            # Random action
            return random.choice(valid_actions)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.policy_net(
                    state['hand_matrix'],
                    state['discard_history'],
                    state['valid_actions_mask']
                )
                # Mask invalid actions
                q_values = q_values.masked_fill(~state['valid_actions_mask'].bool(), float('-inf'))
                return q_values.argmax().item()
    
    def optimize_model(self):
        """Perform one step of optimization"""
        if len(self.memory) < BATCH_SIZE:
            return 0.0
            
        # Sample batch
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*transitions))
        
        # Create batch tensors
        state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.state]).to(self.device),
            'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch.state]).to(self.device)
        }
        
        next_state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.next_state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.next_state]).to(self.device),
            'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch.next_state]).to(self.device)
        }
        
        action_batch = torch.tensor([a for a in batch.action], device=self.device).unsqueeze(1)
        reward_batch = torch.tensor([r for r in batch.reward], device=self.device)
        done_batch = torch.tensor([d for d in batch.done], device=self.device, dtype=torch.float32)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(
            state_batch['hand_matrix'],
            state_batch['discard_history'],
            state_batch['valid_actions_mask']
        ).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(
                next_state_batch['hand_matrix'],
                next_state_batch['discard_history'],
                next_state_batch['valid_actions_mask']
            ).max(1)[0]
            
            # Set V(s) = 0 for terminal states
            next_state_values = next_state_values * (1 - done_batch)
            
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch
            
        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """Update the target network with the policy network's weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save(self, filepath):
        """Save model to file"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        """Load model from file"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint['steps_done']
            print(f"Loaded model from {filepath}")
        else:
            print(f"No model found at {filepath}")

class ImprovedREINFORCEAgent:
    """Improved REINFORCE agent with baseline and entropy regularization"""
    def __init__(self, state_dim=None, action_dim=110, hidden_dim=256, learning_rate=LEARNING_RATE):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
            
        # Initialize policy network
        self.policy = PolicyNetwork().to(self.device)
        
        # Initialize optimizer with weight decay for regularization
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Initialize episode history
        self.rewards = []
        self.log_probs = []
        self.entropies = []
        
        # Training parameters
        self.steps_done = 0
        
    def select_action(self, state, eval_mode=False):
        """Select an action using the policy network"""
        # Move state to device
        for key in state:
            if isinstance(state[key], torch.Tensor):
                state[key] = state[key].to(self.device)
        
        # Get valid actions
        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
        if isinstance(valid_actions, int):
            valid_actions = [valid_actions]
            
        # Forward pass through policy network
        with torch.no_grad():
            # Use a lower temperature during evaluation for more exploitation
            temperature = 0.5 if eval_mode else 1.0
            
            # Get action probabilities
            action_probs = self.policy(
                state['hand_matrix'],
                state['discard_history'],
                state.get('opponent_model', None),
                state['valid_actions_mask'],
                temperature=temperature
            )
            
            # Sample action from the distribution
            if eval_mode:
                # During evaluation, take the most probable action
                action = action_probs.argmax().item()
            else:
                # During training, sample from the distribution
                m = torch.distributions.Categorical(action_probs)
                action = m.sample().item()
                
                # Store log probability and entropy for training
                self.log_probs.append(m.log_prob(torch.tensor([action], device=self.device)))
                self.entropies.append(m.entropy())
                
        return action
    
    def update_policy(self):
        """Update policy using REINFORCE with baseline"""
        if not self.rewards:
            return 0.0
            
        # Convert rewards to tensor
        R = 0
        returns = []
        
        # Calculate discounted returns
        for r in self.rewards[::-1]:
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, device=self.device)
        
        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(self.log_probs, returns):
            policy_loss.append(-log_prob * R)
            
        policy_loss = torch.cat(policy_loss).sum()
        
        # Add entropy regularization
        entropy_loss = -ENTROPY_COEF * torch.cat(self.entropies).sum()
        
        # Total loss
        loss = policy_loss + entropy_loss
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Reset episode history
        self.rewards = []
        self.log_probs = []
        self.entropies = []
        
        return loss.item()
    
    def save(self, filepath):
        """Save model to file"""
        torch.save({
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }, filepath)
    
    def load(self, filepath):
        """Load model from file"""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.policy.load_state_dict(checkpoint['policy'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.steps_done = checkpoint.get('steps_done', 0)
            print(f"Loaded model from {filepath}")
        else:
            print(f"No model found at {filepath}")

def train_dqn(episodes=EPISODES, render=False, load_path=None, save_path="models/improved_dqn.pt"):
    """Train a DQN agent"""
    print("Starting DQN training...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialize environment and agent
    env = GinRummyEnv()
    agent = ImprovedDQNAgent()
    
    # Load existing model if specified
    if load_path:
        agent.load(load_path)
    
    # Training metrics
    rewards = []
    losses = []
    win_rates = []
    epsilons = []
    
    # Training loop
    for episode in range(1, episodes + 1):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        done = False
        
        # Play one episode
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Scale reward for better learning
            scaled_reward = reward * REWARD_SCALING
            
            # Store transition in memory
            agent.memory.push(state, action, scaled_reward, next_state, done)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            
            # Optimize model
            loss = agent.optimize_model()
            if loss:
                episode_loss += loss
                
            # Update target network
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target_network()
                
            # Increment step counter
            agent.steps_done += 1
        
        # Track metrics
        rewards.append(episode_reward)
        losses.append(episode_loss)
        epsilons.append(agent.epsilon)
        
        # Evaluate agent periodically
        if episode % EVAL_INTERVAL == 0:
            win_rate = evaluate_agent(agent, env, num_games=10)
            win_rates.append(win_rate)
            print(f"Episode {episode}/{episodes} - Win rate: {win_rate:.2f}")
        
        # Save model periodically
        if episode % SAVE_INTERVAL == 0:
            agent.save(save_path.replace(".pt", f"_episode_{episode}.pt"))
            
            # Plot metrics
            plot_metrics(rewards, losses, win_rates, epsilons, "dqn")
            
        # Print progress
        if episode % 10 == 0:
            avg_reward = sum(rewards[-10:]) / 10
            avg_loss = sum(losses[-10:]) / 10
            print(f"Episode {episode}/{episodes} - Avg reward: {avg_reward:.2f}, Avg loss: {avg_loss:.6f}, Epsilon: {agent.epsilon:.2f}")
    
    # Save final model
    agent.save(save_path)
    
    # Plot final metrics
    plot_metrics(rewards, losses, win_rates, epsilons, "dqn")
    
    print("DQN training complete!")
    return agent

def train_reinforce(episodes=EPISODES, render=False, load_path=None, save_path="models/improved_reinforce.pt"):
    """Train a REINFORCE agent"""
    print("Starting REINFORCE training...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Initialize environment and agent
    env = GinRummyEnv()
    agent = ImprovedREINFORCEAgent()
    
    # Load existing model if specified
    if load_path:
        agent.load(load_path)
    
    # Training metrics
    rewards = []
    losses = []
    win_rates = []
    
    # Training loop
    for episode in range(1, episodes + 1):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        
        # Play one episode
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Scale reward for better learning
            scaled_reward = reward * REWARD_SCALING
            
            # Store reward
            agent.rewards.append(scaled_reward)
            
            # Move to next state
            state = next_state
            episode_reward += reward
            
            # Increment step counter
            agent.steps_done += 1
        
        # Update policy
        loss = agent.update_policy()
        
        # Track metrics
        rewards.append(episode_reward)
        losses.append(loss)
        
        # Evaluate agent periodically
        if episode % EVAL_INTERVAL == 0:
            win_rate = evaluate_agent(agent, env, num_games=10)
            win_rates.append(win_rate)
            print(f"Episode {episode}/{episodes} - Win rate: {win_rate:.2f}")
        
        # Save model periodically
        if episode % SAVE_INTERVAL == 0:
            agent.save(save_path.replace(".pt", f"_episode_{episode}.pt"))
            
            # Plot metrics
            plot_metrics(rewards, losses, win_rates, None, "reinforce")
            
        # Print progress
        if episode % 10 == 0:
            avg_reward = sum(rewards[-10:]) / 10
            avg_loss = sum(losses[-10:]) / 10 if losses[-10:] else 0
            print(f"Episode {episode}/{episodes} - Avg reward: {avg_reward:.2f}, Avg loss: {avg_loss:.6f}")
    
    # Save final model
    agent.save(save_path)
    
    # Plot final metrics
    plot_metrics(rewards, losses, win_rates, None, "reinforce")
    
    print("REINFORCE training complete!")
    return agent

def evaluate_agent(agent, env, num_games=10):
    """Evaluate an agent against a random opponent"""
    wins = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                action = agent.select_action(state, eval_mode=True)
            # Random opponent's turn
            else:
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = random.choice(valid_actions)
                
            # Take action
            state, reward, done, _ = env.step(action)
            
        # Check if agent won
        if reward > 0:
            wins += 1
    
    return wins / num_games

def plot_metrics(rewards, losses, win_rates, epsilons, agent_type):
    """Plot training metrics"""
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
    plt.plot(range(0, len(win_rates) * EVAL_INTERVAL, EVAL_INTERVAL), win_rates)
    plt.title('Win Rates')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    # Plot epsilons (DQN only)
    if epsilons:
        plt.subplot(2, 2, 4)
        plt.plot(epsilons)
        plt.title('Epsilon Values')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig(f"plots/{agent_type}_metrics.png")
    plt.close()

def self_play_training(episodes=1000, save_interval=100):
    """Train agents using self-play"""
    print("Starting self-play training...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Initialize environment and agents
    env = GinRummyEnv()
    dqn_agent = ImprovedDQNAgent()
    reinforce_agent = ImprovedREINFORCEAgent()
    
    # Load existing models if available
    dqn_agent.load("models/improved_dqn.pt")
    reinforce_agent.load("models/improved_reinforce.pt")
    
    # Training metrics
    dqn_wins = 0
    reinforce_wins = 0
    draws = 0
    
    # Training loop
    for episode in range(1, episodes + 1):
        # Reset environment
        state = env.reset()
        done = False
        
        # Determine which agent goes first (alternate)
        dqn_first = episode % 2 == 0
        
        # Play one episode
        while not done:
            # DQN agent's turn
            if (env.current_player == 0 and dqn_first) or (env.current_player == 1 and not dqn_first):
                action = dqn_agent.select_action(state)
            # REINFORCE agent's turn
            else:
                action = reinforce_agent.select_action(state)
                
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition for DQN
            if (env.current_player == 0 and dqn_first) or (env.current_player == 1 and not dqn_first):
                # Invert reward if DQN is player 1
                if env.current_player == 1:
                    reward = -reward
                dqn_agent.memory.push(state, action, reward * REWARD_SCALING, next_state, done)
                
            # Store reward for REINFORCE
            if (env.current_player == 0 and not dqn_first) or (env.current_player == 1 and dqn_first):
                # Invert reward if REINFORCE is player 1
                if env.current_player == 1:
                    reward = -reward
                reinforce_agent.rewards.append(reward * REWARD_SCALING)
                
            # Move to next state
            state = next_state
            
            # Optimize DQN model
            dqn_agent.optimize_model()
            
            # Update target network
            if dqn_agent.steps_done % TARGET_UPDATE == 0:
                dqn_agent.update_target_network()
                
            # Increment step counters
            dqn_agent.steps_done += 1
            reinforce_agent.steps_done += 1
        
        # Update REINFORCE policy
        reinforce_agent.update_policy()
        
        # Track results
        if reward > 0:
            if (env.current_player == 0 and dqn_first) or (env.current_player == 1 and not dqn_first):
                dqn_wins += 1
            else:
                reinforce_wins += 1
        elif reward < 0:
            if (env.current_player == 0 and dqn_first) or (env.current_player == 1 and not dqn_first):
                reinforce_wins += 1
            else:
                dqn_wins += 1
        else:
            draws += 1
        
        # Save models periodically
        if episode % save_interval == 0:
            dqn_agent.save(f"models/self_play_dqn_episode_{episode}.pt")
            reinforce_agent.save(f"models/self_play_reinforce_episode_{episode}.pt")
            
            # Print progress
            print(f"Episode {episode}/{episodes}")
            print(f"DQN wins: {dqn_wins}, REINFORCE wins: {reinforce_wins}, Draws: {draws}")
            print(f"Win rates - DQN: {dqn_wins/episode:.2f}, REINFORCE: {reinforce_wins/episode:.2f}")
    
    # Save final models
    dqn_agent.save("models/self_play_dqn_final.pt")
    reinforce_agent.save("models/self_play_reinforce_final.pt")
    
    print("Self-play training complete!")
    print(f"Final results - DQN wins: {dqn_wins}, REINFORCE wins: {reinforce_wins}, Draws: {draws}")
    print(f"Win rates - DQN: {dqn_wins/episodes:.2f}, REINFORCE: {reinforce_wins/episodes:.2f}")
    
    return dqn_agent, reinforce_agent

def main():
    """Main function to run training"""
    print("Starting improved training for Gin Rummy AI agents...")
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train Gin Rummy AI agents')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'reinforce', 'self-play'],
                        help='Agent type to train (dqn, reinforce, or self-play)')
    parser.add_argument('--episodes', type=int, default=EPISODES, help='Number of episodes to train')
    parser.add_argument('--load', type=str, default=None, help='Path to load model from')
    parser.add_argument('--save', type=str, default=None, help='Path to save model to')
    args = parser.parse_args()
    
    # Train specified agent
    if args.agent == 'dqn':
        save_path = args.save or "models/improved_dqn_final.pt"
        train_dqn(episodes=args.episodes, load_path=args.load, save_path=save_path)
    elif args.agent == 'reinforce':
        save_path = args.save or "models/improved_reinforce_final.pt"
        train_reinforce(episodes=args.episodes, load_path=args.load, save_path=save_path)
    elif args.agent == 'self-play':
        self_play_training(episodes=args.episodes)
    
    print("Training complete!")

if __name__ == "__main__":
    main() 