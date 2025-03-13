#!/usr/bin/env python3

import os
import argparse
import torch
import random
import numpy as np
from tqdm import tqdm

# Import our models and environment
from dqn import DQNAgent, DQNetwork
from reinforce import REINFORCEAgent, PolicyNetwork
from mcts import MCTSAgent, PolicyValueNetwork
from gin_rummy_env import GinRummyEnv
from improved_training import ImprovedDQNAgent, ImprovedREINFORCEAgent, train_dqn, train_reinforce, self_play_training

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

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Train and evaluate Gin Rummy AI agents")
    parser.add_argument("--train", action="store_true", help="Train models")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate models")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes for training")
    parser.add_argument("--games", type=int, default=50, help="Number of games for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Print detailed output")
    
    args = parser.parse_args()
    
    if args.train:
        train_all_models(episodes=args.episodes)
    
    if args.evaluate:
        evaluate_models(num_games=args.games, verbose=args.verbose)
    
    if not args.train and not args.evaluate:
        print("Please specify --train or --evaluate")

if __name__ == "__main__":
    main() 