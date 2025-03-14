#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import time
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN
from improved_quick_train import PolicyValueNetwork, MCTSAgent, evaluate_agent

def train_mcts(num_episodes=5000, lr=0.0001, num_simulations=50, c_puct=1.0, 
              reward_shaping=True, deadwood_reward_scale=0.03, win_reward=2.0, 
              gin_reward=3.0, knock_reward=1.0, eval_interval=500, 
              save_interval=1000, model_path="models/improved_mcts.pt"):
    """Train an MCTS agent for Gin Rummy."""
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
    
    # Set up policy value network and optimizer
    policy_value_net = PolicyValueNetwork().to(device)
    optimizer = optim.Adam(policy_value_net.parameters(), lr=lr)
    
    # Set up MCTS agent
    mcts_agent = MCTSAgent(policy_value_net, device, num_simulations=num_simulations, c_puct=c_puct)
    
    # Training metrics
    episode_rewards = []
    win_rates = []
    policy_losses = []
    value_losses = []
    
    # Create directories for saving models and metrics
    os.makedirs("models", exist_ok=True)
    os.makedirs("metrics", exist_ok=True)
    
    # Progress bar
    pbar = tqdm(range(num_episodes), desc="Training MCTS")
    
    for episode in pbar:
        # Reset environment and MCTS tree
        state = env.reset()
        mcts_agent.reset()
        
        # Training data
        states = []
        mcts_probs = []
        values = []
        
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
                
                # Get action probabilities from MCTS
                action_probs = mcts_agent._get_action_probs(state_device, valid_actions, temp=1.0)
                
                # Store state and MCTS probabilities
                states.append(state_device)
                mcts_probs.append(action_probs)
                
                # Create a mask for valid actions only
                valid_action_mask = torch.zeros_like(action_probs)
                for a in valid_actions:
                    valid_action_mask[a] = 1.0
                
                # Apply mask to ensure we only sample from valid actions
                masked_probs = action_probs * valid_action_mask
                masked_probs = masked_probs / (masked_probs.sum() + 1e-8)
                
                # Sample action from MCTS probabilities
                action_idx = torch.multinomial(masked_probs, 1).item()
                action = action_idx  # The action is the index itself
            else:
                # Opponent's turn - random action
                action = np.random.choice(valid_actions)
            
            # Take action
            next_state, reward, done, _, info = env.step(action)
            
            # Store reward if it's the agent's turn
            if env.current_player == 0:
                episode_reward += reward
            
            # Update state
            state = next_state
        
        # Store final outcome
        if 'winner' in info:
            winner = info['winner']
        elif 'outcome' in info:
            # Determine winner based on outcome and current player
            if info['outcome'] == 'win' or info['outcome'] == 'gin':
                winner = 1 - env.current_player  # The player who just played won
            else:
                winner = env.current_player  # The player who just played lost
        else:
            # Default to a draw if no winner information is available
            winner = -1
            
        outcome = 1.0 if winner == 0 else -1.0
        values = [torch.tensor([outcome], dtype=torch.float32, device=device) for _ in range(len(states))]
        
        # Convert to tensors
        mcts_probs = torch.stack(mcts_probs)
        
        # Train on episode data
        episode_policy_loss = 0.0
        episode_value_loss = 0.0
        
        for i in range(0, len(states), 32):  # Process in batches of 32
            batch_states = states[i:i+32]
            batch_mcts_probs = mcts_probs[i:i+32]
            batch_values = values[i:i+32]
            
            if len(batch_states) == 0:
                continue
            
            # Prepare batch
            batch_hand_matrix = torch.stack([s['hand_matrix'] for s in batch_states])
            batch_discard_history = torch.stack([s['discard_history'] for s in batch_states])
            
            # Forward pass
            policy_logits, value_preds = policy_value_net(batch_hand_matrix, batch_discard_history)
            
            # Calculate loss
            policy_loss = -torch.mean(torch.sum(batch_mcts_probs * torch.log(policy_logits + 1e-8), dim=1))
            value_loss = torch.mean(torch.pow(value_preds - torch.cat(batch_values), 2))
            total_loss = policy_loss + value_loss
            
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Track losses
            episode_policy_loss += policy_loss.item()
            episode_value_loss += value_loss.item()
        
        # Average losses for the episode
        if len(states) > 0:
            episode_policy_loss /= (len(states) + 31) // 32  # Number of batches
            episode_value_loss /= (len(states) + 31) // 32
        
        # Track metrics
        episode_rewards.append(episode_reward)
        policy_losses.append(episode_policy_loss)
        value_losses.append(episode_value_loss)
        
        # Evaluate agent
        if (episode + 1) % eval_interval == 0:
            win_rate = evaluate_mcts_agent(policy_value_net, env, device, num_simulations, num_games=20)[0]
            win_rates.append(win_rate)
            
            # Update progress bar
            pbar.set_postfix({
                'win_rate': f'{win_rate:.2f}',
                'reward': f'{episode_reward:.2f}',
                'p_loss': f'{episode_policy_loss:.4f}',
                'v_loss': f'{episode_value_loss:.4f}'
            })
            
            # Save metrics
            metrics = {
                'episode_rewards': episode_rewards,
                'policy_losses': policy_losses,
                'value_losses': value_losses,
                'win_rates': win_rates,
                'eval_episodes': list(range(eval_interval, episode + 2, eval_interval))
            }
            
            with open('metrics/mcts_training_metrics.json', 'w') as f:
                json.dump(metrics, f)
            
            # Plot metrics
            plot_metrics(episode_rewards, policy_losses, value_losses, win_rates, eval_interval)
        
        # Save model
        if (episode + 1) % save_interval == 0 and args.save:
            torch.save(policy_value_net.state_dict(), model_path)
            print(f"Model saved to {model_path}")
    
    # Save final model
    if args.save:
        torch.save(policy_value_net.state_dict(), model_path)
        print(f"Final model saved to {model_path}")
    
    return policy_value_net, episode_rewards, win_rates

def evaluate_mcts_agent(policy_value_net, env, device, num_simulations, num_games=20, verbose=False):
    """Evaluate an MCTS agent."""
    policy_value_net.eval()
    
    # Create MCTS agent
    mcts_agent = MCTSAgent(policy_value_net, device, num_simulations=num_simulations)
    
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
                    # Use MCTS to select action
                    action = mcts_agent.select_action(state, valid_actions, training=False)
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
    
    policy_value_net.train()
    return wins / num_games, total_reward / num_games

def plot_metrics(rewards, policy_losses, value_losses, win_rates, eval_interval):
    """Plot training metrics."""
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot rewards
    plt.subplot(2, 2, 1)
    plt.plot(rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot policy losses
    plt.subplot(2, 2, 2)
    plt.plot(policy_losses)
    plt.title('Policy Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot value losses
    plt.subplot(2, 2, 3)
    plt.plot(value_losses)
    plt.title('Value Losses')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot win rates
    plt.subplot(2, 2, 4)
    plt.plot(range(eval_interval, len(win_rates) * eval_interval + 1, eval_interval), win_rates, marker='o')
    plt.title('Win Rates')
    plt.xlabel('Episode')
    plt.ylabel('Win Rate')
    
    plt.tight_layout()
    plt.savefig("plots/mcts_training_metrics.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an MCTS agent for Gin Rummy')
    parser.add_argument('--episodes', type=int, default=5000, help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--simulations', type=int, default=50, help='Number of MCTS simulations per move')
    parser.add_argument('--c-puct', type=float, default=1.0, help='Exploration constant for PUCT')
    parser.add_argument('--reward-shaping', action='store_true', help='Use reward shaping')
    parser.add_argument('--deadwood-reward-scale', type=float, default=0.03, help='Scale for deadwood reduction reward')
    parser.add_argument('--win-reward', type=float, default=2.0, help='Reward for winning')
    parser.add_argument('--gin-reward', type=float, default=3.0, help='Additional reward for gin')
    parser.add_argument('--knock-reward', type=float, default=1.0, help='Additional reward for knock')
    parser.add_argument('--eval-interval', type=int, default=500, help='Interval for evaluation')
    parser.add_argument('--save-interval', type=int, default=1000, help='Interval for saving model')
    parser.add_argument('--save', action='store_true', help='Save model')
    parser.add_argument('--model-path', type=str, default='models/improved_mcts.pt', help='Path to save model')
    
    args = parser.parse_args()
    
    train_mcts(
        num_episodes=args.episodes,
        lr=args.lr,
        num_simulations=args.simulations,
        c_puct=args.c_puct,
        reward_shaping=args.reward_shaping,
        deadwood_reward_scale=args.deadwood_reward_scale,
        win_reward=args.win_reward,
        gin_reward=args.gin_reward,
        knock_reward=args.knock_reward,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        model_path=args.model_path
    ) 