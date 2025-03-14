#!/usr/bin/env python3

import os
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Import our models and environment
from dqn import DQNAgent, DQNetwork
from reinforce import REINFORCEAgent, PolicyNetwork
from mcts import MCTSAgent, PolicyValueNetwork
from gin_rummy_env import GinRummyEnv
from improved_training import ImprovedDQNAgent, ImprovedREINFORCEAgent, train_dqn, train_reinforce, self_play_training
from rules_based_agent import RulesBasedAgent
from improved_gin_rummy_env import ImprovedGinRummyEnv

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

def evaluate_models(num_games=50, verbose=False):
    """Evaluate all models against random and against each other"""
    print("\n=== Evaluating Models ===\n")
    
    # Create environment
    env = GinRummyEnv()
    
    # Create agents
    dqn_agent = ImprovedDQNAgent()
    reinforce_agent = ImprovedREINFORCEAgent()
    
    # Load models if available
    dqn_agent.load("models/improved_dqn_final.pt")
    reinforce_agent.load("models/improved_reinforce_final.pt")
    
    # Create a random agent
    class RandomAgent:
        def select_action(self, state, eval_mode=False):
            valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
            return random.choice(valid_actions)
    
    random_agent = RandomAgent()
    
    # Evaluate DQN vs Random
    print("\nEvaluating DQN vs Random...")
    dqn_vs_random_results = evaluate_agent_vs_agent(dqn_agent, random_agent, env, num_games, verbose)
    
    # Evaluate REINFORCE vs Random
    print("\nEvaluating REINFORCE vs Random...")
    reinforce_vs_random_results = evaluate_agent_vs_agent(reinforce_agent, random_agent, env, num_games, verbose)
    
    # Evaluate DQN vs REINFORCE
    print("\nEvaluating DQN vs REINFORCE...")
    dqn_vs_reinforce_results = evaluate_agent_vs_agent(dqn_agent, reinforce_agent, env, num_games, verbose)
    
    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"DQN vs Random: {dqn_vs_random_results}")
    print(f"REINFORCE vs Random: {reinforce_vs_random_results}")
    print(f"DQN vs REINFORCE: {dqn_vs_reinforce_results}")

def evaluate_agent_vs_agent(agent1, agent2, env, num_games=10, verbose=False):
    """Evaluate agent1 vs agent2"""
    agent1_wins = 0
    agent2_wins = 0
    draws = 0
    
    for game in tqdm(range(num_games), desc="Games"):
        state = env.reset()
        done = False
        turn = 0
        
        while not done and turn < 100:  # Limit to 100 turns to prevent infinite games
            if verbose and turn % 10 == 0:
                print(f"\nGame {game+1}, Turn {turn+1}")
                env.print_state()
            
            # Agent 1's turn
            if env.current_player == 0:
                action = agent1.select_action(state, eval_mode=True)
                if verbose:
                    print(f"Agent 1 took action {action}")
            # Agent 2's turn
            else:
                action = agent2.select_action(state, eval_mode=True)
                if verbose:
                    print(f"Agent 2 took action {action}")
            
            # Take action
            state, reward, done, _ = env.step(action)
            turn += 1
        
        # Determine winner
        if reward > 0:
            if env.current_player == 0:
                agent1_wins += 1
                if verbose:
                    print(f"Game {game+1}: Agent 1 wins")
            else:
                agent2_wins += 1
                if verbose:
                    print(f"Game {game+1}: Agent 2 wins")
        elif reward < 0:
            if env.current_player == 0:
                agent2_wins += 1
                if verbose:
                    print(f"Game {game+1}: Agent 2 wins")
            else:
                agent1_wins += 1
                if verbose:
                    print(f"Game {game+1}: Agent 1 wins")
        else:
            draws += 1
            if verbose:
                print(f"Game {game+1}: Draw")
    
    # Calculate win rates
    agent1_win_rate = agent1_wins / num_games
    agent2_win_rate = agent2_wins / num_games
    draw_rate = draws / num_games
    
    return {
        "agent1_wins": agent1_wins,
        "agent2_wins": agent2_wins,
        "draws": draws,
        "agent1_win_rate": agent1_win_rate,
        "agent2_win_rate": agent2_win_rate,
        "draw_rate": draw_rate
    }

def train_all_models(episodes=1000):
    """Train all models"""
    print("\n=== Training Models ===\n")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Train DQN
    print("\nTraining DQN...")
    train_dqn(episodes=episodes, save_path="models/improved_dqn_final.pt")
    
    # Train REINFORCE
    print("\nTraining REINFORCE...")
    train_reinforce(episodes=episodes, save_path="models/improved_reinforce_final.pt")
    
    # Train with self-play
    print("\nTraining with self-play...")
    self_play_training(episodes=episodes)

def train_reinforce(
    env: ImprovedGinRummyEnv,
    agent: REINFORCEAgent,
    num_episodes: int,
    eval_interval: int = 100,
    save_dir: str = "models"
) -> Dict[str, List[float]]:
    """Train REINFORCE agent and track metrics."""
    os.makedirs(save_dir, exist_ok=True)
    
    metrics = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'eval_rewards': []
    }
    
    # Create rules-based opponent for evaluation
    eval_opponent = RulesBasedAgent()
    
    for episode in tqdm(range(num_episodes), desc="Training"):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            
            # Store experience
            agent.store_experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done
            )
            
            state = next_state
            episode_reward += reward
        
        # Train on episode
        losses = agent.train()
        
        # Record metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['policy_losses'].append(losses['policy_loss'])
        metrics['value_losses'].append(losses['value_loss'])
        metrics['entropies'].append(losses['entropy'])
        
        # Evaluate periodically
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_against_rules(agent, eval_opponent, env, num_games=50)
            metrics['eval_rewards'].append(eval_reward)
            
            # Save model
            agent.save(os.path.join(save_dir, f"reinforce_episode_{episode+1}.pt"))
            
            # Plot current progress
            plot_metrics(metrics, episode + 1)
    
    return metrics

def evaluate_against_rules(
    agent: REINFORCEAgent,
    opponent: RulesBasedAgent,
    env: ImprovedGinRummyEnv,
    num_games: int = 100
) -> float:
    """Evaluate agent against rules-based opponent."""
    total_reward = 0
    
    for _ in range(num_games):
        state = env.reset()
        done = False
        
        while not done:
            # Agent's turn
            if env.current_player == 0:
                action = agent.select_action(state)
            else:
                action = opponent.select_action(state)
            
            state, reward, done, truncated, info = env.step(action)
            if env.current_player == 0:  # Only count agent's rewards
                total_reward += reward
    
    return total_reward / num_games

def plot_metrics(metrics: Dict[str, List[float]], episode: int) -> None:
    """Plot training metrics."""
    plt.figure(figsize=(15, 10))
    
    # Plot episode rewards
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    
    # Plot policy loss
    plt.subplot(2, 2, 2)
    plt.plot(metrics['policy_losses'])
    plt.title('Policy Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot value loss
    plt.subplot(2, 2, 3)
    plt.plot(metrics['value_losses'])
    plt.title('Value Loss')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    
    # Plot evaluation rewards
    plt.subplot(2, 2, 4)
    eval_episodes = list(range(100, episode + 1, 100))
    plt.plot(eval_episodes, metrics['eval_rewards'])
    plt.title('Evaluation Rewards vs Rules-Based')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Train and evaluate Gin Rummy agents')
    parser.add_argument('--train', action='store_true', help='Train the agents')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the agents')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes for training')
    parser.add_argument('--eval-games', type=int, default=1000, help='Number of games for evaluation')
    parser.add_argument('--verbose', action='store_true', help='Print detailed output')
    args = parser.parse_args()
    
    if not args.train and not args.evaluate:
        parser.print_help()
        return
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    if args.train:
        # Create environment and agent
        env = ImprovedGinRummyEnv()
        agent = REINFORCEAgent()
        
        # Train agent
        metrics = train_reinforce(
            env=env,
            agent=agent,
            num_episodes=args.episodes,
            eval_interval=100
        )
        
        # Plot final metrics
        plot_metrics(metrics, args.episodes)
    
    if args.evaluate:
        # Create environment and agents
        env = ImprovedGinRummyEnv()
        agent = REINFORCEAgent()
        rules_agent = RulesBasedAgent()
        
        # Load latest model if available
        model_files = [f for f in os.listdir('models') if f.startswith('reinforce_episode_')]
        if model_files:
            latest_model = max(model_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
            print(f"Loading model: {latest_model}")
            agent.load_model(os.path.join('models', latest_model))
        
        # Evaluate against rules-based agent
        final_reward = evaluate_against_rules(
            agent=agent,
            opponent=rules_agent,
            env=env,
            num_games=args.eval_games
        )
        print(f"Final evaluation reward against rules-based agent: {final_reward:.2f}")

if __name__ == "__main__":
    main() 