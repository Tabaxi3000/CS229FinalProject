#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import time
from tqdm import tqdm
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN
from improved_quick_train import PolicyNetwork, evaluate_agent
import random

def train_reinforce(num_episodes=5000, lr=0.0001, gamma=0.99, reward_shaping=True, 
                   deadwood_reward_scale=0.03, win_reward=2.0, gin_reward=3.0, knock_reward=1.0,
                   eval_interval=500, save_interval=1000, model_path="models/improved_reinforce.pt"):
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
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Training statistics
    episode_rewards = []
    win_rates = []
    best_win_rate = 0.0
    
    # Training loop
    start_time = time.time()
    
    for episode in tqdm(range(num_episodes), desc="Training REINFORCE"):
        # Reset environment
        state = env.reset()
        done = False
        
        # Lists to store episode data
        log_probs = []
        rewards = []
        
        # Play one episode
        while not done:
            # Check if it's the agent's turn
            if env.current_player == 0:
                # Get valid actions
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                
                # Prioritize GIN and KNOCK actions
                if GIN in valid_actions:
                    action = GIN
                    env.gin_taken += 1
                    env.gin_opportunities += 1
                elif KNOCK in valid_actions:
                    action = KNOCK
                    env.knock_taken += 1
                    env.knock_opportunities += 1
                else:
                    # Move state to device
                    state_device = {
                        'hand_matrix': state['hand_matrix'].to(device),
                        'discard_history': state['discard_history'].to(device),
                        'valid_actions_mask': state['valid_actions_mask'].to(device)
                    }
                    
                    # Get action probabilities from policy network
                    action_probs = policy_net(
                        state_device['hand_matrix'],
                        state_device['discard_history']
                    ).squeeze()
                    
                    # Apply mask to action probabilities
                    masked_probs = action_probs * state_device['valid_actions_mask']
                    masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
                    
                    # Sample action from distribution
                    m = torch.distributions.Categorical(masked_probs)
                    action = m.sample().item()
                    
                    # Store log probability for training
                    log_probs.append(m.log_prob(torch.tensor(action, device=device)))
                    
                    # Check for GIN/KNOCK opportunities
                    if GIN in valid_actions:
                        env.gin_opportunities += 1
                    if KNOCK in valid_actions:
                        env.knock_opportunities += 1
            else:
                # Opponent's turn - random action
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action = np.random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Store reward if it's the agent's turn
            if env.current_player == 1:  # Agent just took action
                rewards.append(reward)
            
            # Update state
            state = next_state
        
        # Calculate episode return
        episode_return = sum(rewards)
        episode_rewards.append(episode_return)
        
        # Calculate discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        if policy_loss:
            if len(policy_loss) == 1:
                policy_loss = policy_loss[0]
            else:
                policy_loss = torch.stack(policy_loss).sum()
            
            # Optimize
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
        
        # Evaluate and save model periodically
        if (episode + 1) % eval_interval == 0:
            # Evaluate agent
            win_rate, avg_reward = evaluate_agent(policy_net, env, device, num_games=20, verbose=False)
            win_rates.append(win_rate)
            
            print(f"\nEpisode {episode+1}/{num_episodes}")
            print(f"  Win rate: {win_rate*100:.1f}%")
            print(f"  Average reward: {avg_reward:.4f}")
            print(f"  Average episode return: {np.mean(episode_rewards[-eval_interval:]):.4f}")
            print(f"  Time elapsed: {time.time() - start_time:.1f} seconds")
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                torch.save(policy_net.state_dict(), model_path.replace(".pt", "_best.pt"))
                print(f"  New best model saved with win rate {win_rate*100:.1f}%")
            
            # Reset opportunity counters
            env.gin_opportunities = 0
            env.knock_opportunities = 0
            env.gin_taken = 0
            env.knock_taken = 0
        
        # Save model periodically
        if (episode + 1) % save_interval == 0:
            torch.save(policy_net.state_dict(), model_path)
            print(f"\nModel saved at episode {episode+1}")
    
    # Save final model
    torch.save(policy_net.state_dict(), model_path.replace(".pt", "_final.pt"))
    print(f"\nFinal model saved after {num_episodes} episodes")
    print(f"Best win rate: {best_win_rate*100:.1f}%")
    print(f"Training time: {time.time() - start_time:.1f} seconds")
    
    return policy_net, episode_rewards, win_rates

def evaluate_agent(policy_net, env, device, num_games=20, verbose=False):
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
                        action_probs = policy_net(
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
                action = np.random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a REINFORCE agent for Gin Rummy')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--reward-shaping', action='store_true', help='Use reward shaping')
    parser.add_argument('--deadwood-reward-scale', type=float, default=0.03, help='Scale factor for deadwood reward')
    parser.add_argument('--win-reward', type=float, default=2.0, help='Reward for winning')
    parser.add_argument('--gin-reward', type=float, default=3.0, help='Reward for gin')
    parser.add_argument('--knock-reward', type=float, default=1.0, help='Reward for knocking')
    parser.add_argument('--eval-interval', type=int, default=500, help='Interval for evaluation')
    parser.add_argument('--save-interval', type=int, default=1000, help='Interval for saving model')
    parser.add_argument('--model-path', type=str, default='models/improved_reinforce.pt', help='Path to save model')
    parser.add_argument('--save', action='store_true', help='Save model')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Train the agent
    train_reinforce(
        num_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        reward_shaping=args.reward_shaping,
        deadwood_reward_scale=args.deadwood_reward_scale,
        win_reward=args.win_reward,
        gin_reward=args.gin_reward,
        knock_reward=args.knock_reward,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        model_path=args.model_path
    ) 