#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import argparse
import os
import time
from tqdm import tqdm
from collections import deque
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN
from improved_quick_train import DQNetwork, evaluate_agent

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4, beta_increment=0.001):
        self.capacity = capacity
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling weight
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def add(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
            self.size += 1
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        if self.size < batch_size:
            indices = np.random.choice(self.size, batch_size, replace=True)
        else:
            priorities = self.priorities[:self.size]
            probabilities = priorities ** self.alpha
            probabilities /= probabilities.sum()
            indices = np.random.choice(self.size, batch_size, p=probabilities, replace=False)
        
        # Importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            self.priorities[idx] = error + 1e-5  # Small constant to ensure non-zero priority

class ImprovedDQNAgent:
    def __init__(self, state_dim, action_dim, device, 
                 learning_rate=0.0001, gamma=0.99, 
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
                 buffer_size=100000, batch_size=64, target_update=10,
                 double_dqn=True, dueling_dqn=False, noisy_net=False,
                 prioritized_replay=True):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.double_dqn = double_dqn
        self.dueling_dqn = dueling_dqn
        self.noisy_net = noisy_net
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Networks
        self.policy_net = DQNetwork().to(device)
        self.target_net = DQNetwork().to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        if prioritized_replay:
            self.memory = PrioritizedReplayBuffer(buffer_size)
            self.prioritized_replay = True
        else:
            self.memory = deque(maxlen=buffer_size)
            self.prioritized_replay = False
        
        # Training info
        self.steps_done = 0
        self.episode_rewards = []
    
    def select_action(self, state, valid_actions, training=True):
        # Prioritize GIN and KNOCK actions
        if GIN in valid_actions:
            return GIN
        elif KNOCK in valid_actions:
            return KNOCK
        
        # Epsilon-greedy action selection
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        else:
            with torch.no_grad():
                # Move state to device
                state_device = {
                    'hand_matrix': state['hand_matrix'].to(self.device),
                    'discard_history': state['discard_history'].to(self.device),
                    'valid_actions_mask': state['valid_actions_mask'].to(self.device)
                }
                
                # Get Q-values
                q_values = self.policy_net(
                    state_device['hand_matrix'],
                    state_device['discard_history']
                ).squeeze()
                
                # Apply mask to Q-values
                masked_q_values = q_values.clone()
                masked_q_values[~state_device['valid_actions_mask'].bool()] = float('-inf')
                
                # Select best action
                return masked_q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        if self.prioritized_replay:
            self.memory.add(state, action, reward, next_state, done)
        else:
            self.memory.append((state, action, reward, next_state, done))
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def optimize_model(self):
        if self.prioritized_replay:
            if self.memory.size < self.batch_size:
                return 0.0
            
            # Sample from prioritized replay buffer
            states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            if len(self.memory) < self.batch_size:
                return 0.0
            
            # Sample from replay buffer
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            weights = torch.ones(self.batch_size, dtype=torch.float32).to(self.device)
        
        # Process batch
        hand_matrices = torch.cat([s['hand_matrix'] for s in states]).to(self.device)
        discard_histories = torch.cat([s['discard_history'] for s in states]).to(self.device)
        next_hand_matrices = torch.cat([s['hand_matrix'] for s in next_states]).to(self.device)
        next_discard_histories = torch.cat([s['discard_history'] for s in next_states]).to(self.device)
        next_valid_actions_masks = torch.cat([s['valid_actions_mask'] for s in next_states]).to(self.device)
        
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Compute Q(s_t, a)
        q_values = self.policy_net(hand_matrices, discard_histories).squeeze()
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute Q(s_{t+1}, a) for all a
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: use policy net to select action, target net to evaluate
                next_q_values = self.policy_net(next_hand_matrices, next_discard_histories).squeeze()
                next_q_values[~next_valid_actions_masks.bool()] = float('-inf')
                next_actions = next_q_values.argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_hand_matrices, next_discard_histories).squeeze()
                next_q_values = next_q_values.gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_hand_matrices, next_discard_histories).squeeze()
                next_q_values[~next_valid_actions_masks.bool()] = float('-inf')
                next_q_values = next_q_values.max(dim=1)[0]
            
            # Compute target Q value
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        td_errors = torch.abs(q_values - target_q_values).detach().cpu().numpy()
        loss = (weights * (q_values - target_q_values) ** 2).mean()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        
        # Update priorities in replay buffer
        if self.prioritized_replay:
            self.memory.update_priorities(indices, td_errors)
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())

def train_dqn(num_episodes=5000, lr=0.0001, gamma=0.99, 
             epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995,
             buffer_size=100000, batch_size=64, target_update=10,
             double_dqn=True, dueling_dqn=False, noisy_net=False,
             prioritized_replay=True, reward_shaping=True, 
             deadwood_reward_scale=0.03, win_reward=2.0, gin_reward=3.0, knock_reward=1.0,
             eval_interval=500, save_interval=1000, model_path="models/improved_dqn.pt"):
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
    
    # Set up agent
    state_dim = (4, 13, 4)  # Hand matrix shape
    action_dim = env.action_space.n
    agent = ImprovedDQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        learning_rate=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        target_update=target_update,
        double_dqn=double_dqn,
        dueling_dqn=dueling_dqn,
        noisy_net=noisy_net,
        prioritized_replay=prioritized_replay
    )
    
    # Training metrics
    episode_rewards = []
    win_rates = []
    losses = []
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training DQN")
    
    for episode in pbar:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        num_updates = 0
        
        # Play one episode
        done = False
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
                
                # Select action
                action = agent.select_action(state_device, valid_actions)
            else:
                # Opponent's turn - random action
                action = random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store transition if it's the agent's turn
            if env.current_player == 0:
                # Move next_state to device
                next_state_device = {
                    'hand_matrix': next_state['hand_matrix'].to(device),
                    'discard_history': next_state['discard_history'].to(device),
                    'valid_actions_mask': next_state['valid_actions_mask'].to(device)
                }
                
                # Store transition
                agent.store_transition(state_device, action, reward, next_state_device, done)
                
                # Update agent
                if len(agent.memory) > batch_size:
                    loss = agent.optimize_model()
                    episode_loss += loss
                    num_updates += 1
                
                # Track reward
                episode_reward += reward
            
            # Update state
            state = next_state
        
        # Update target network
        if (episode + 1) % target_update == 0:
            agent.update_target_network()
        
        # Update epsilon
        agent.update_epsilon()
        
        # Track metrics
        episode_rewards.append(episode_reward)
        if num_updates > 0:
            losses.append(episode_loss / num_updates)
        else:
            losses.append(0)
        
        # Evaluate agent
        if (episode + 1) % eval_interval == 0:
            win_rate = evaluate_agent(agent.policy_net, env, device, num_games=20)[0]
            win_rates.append(win_rate)
            
            # Update progress bar
            pbar.set_postfix({
                'win_rate': f'{win_rate:.2f}',
                'reward': f'{episode_reward:.2f}',
                'loss': f'{losses[-1]:.4f}',
                'epsilon': f'{agent.epsilon:.2f}'
            })
        
        # Save model
        if (episode + 1) % save_interval == 0 and args.save:
            agent.save_model(model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    if args.save:
        agent.save_model(model_path)
        print(f"Final model saved to {model_path}")
    
    return agent.policy_net, episode_rewards, win_rates, losses

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train improved DQN agent for Gin Rummy')
    parser.add_argument('--episodes', type=int, default=5000,
                        help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Starting epsilon for exploration')
    parser.add_argument('--epsilon-end', type=float, default=0.05,
                        help='Minimum epsilon for exploration')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate')
    parser.add_argument('--buffer-size', type=int, default=100000,
                        help='Replay buffer size')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--target-update', type=int, default=10,
                        help='Target network update frequency')
    parser.add_argument('--double-dqn', action='store_true',
                        help='Use Double DQN')
    parser.add_argument('--dueling-dqn', action='store_true',
                        help='Use Dueling DQN')
    parser.add_argument('--noisy-net', action='store_true',
                        help='Use Noisy Networks for exploration')
    parser.add_argument('--prioritized-replay', action='store_true',
                        help='Use Prioritized Experience Replay')
    parser.add_argument('--reward-shaping', action='store_true',
                        help='Use reward shaping')
    parser.add_argument('--deadwood-reward-scale', type=float, default=0.03,
                        help='Scale factor for deadwood reward')
    parser.add_argument('--win-reward', type=float, default=2.0,
                        help='Reward for winning')
    parser.add_argument('--gin-reward', type=float, default=3.0,
                        help='Reward for gin')
    parser.add_argument('--knock-reward', type=float, default=1.0,
                        help='Reward for knock')
    parser.add_argument('--eval-interval', type=int, default=500,
                        help='Evaluation interval')
    parser.add_argument('--save-interval', type=int, default=1000,
                        help='Save interval')
    parser.add_argument('--model-path', type=str, default="models/improved_dqn.pt",
                        help='Path to save model')
    
    args = parser.parse_args()
    
    train_dqn(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update,
        double_dqn=args.double_dqn,
        dueling_dqn=args.dueling_dqn,
        noisy_net=args.noisy_net,
        prioritized_replay=args.prioritized_replay,
        reward_shaping=args.reward_shaping,
        deadwood_reward_scale=args.deadwood_reward_scale,
        win_reward=args.win_reward,
        gin_reward=args.gin_reward,
        knock_reward=args.knock_reward,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        model_path=args.model_path
    ) 