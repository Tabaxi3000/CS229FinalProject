#!/usr/bin/env python3

import torch
import random
import numpy as np
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN
from improved_quick_train import DQNetwork, PolicyNetwork, PolicyValueNetwork

def evaluate_agent(agent_type='dqn', model_path=None, num_games=100, verbose=True):
    """
    Evaluate an agent against a random opponent.
    
    Args:
        agent_type: Type of agent to evaluate ('dqn', 'reinforce', or 'mcts')
        model_path: Path to the model file
        num_games: Number of games to play
        verbose: Whether to print detailed results
    
    Returns:
        win_rate: Percentage of games won
        avg_reward: Average reward per game
    """
    # Set up environment
    env = ImprovedGinRummyEnv(reward_shaping=False)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    # Load model
    if model_path is None:
        if agent_type == 'dqn':
            model_path = 'models/improved_dqn_final.pt'
        elif agent_type == 'reinforce':
            model_path = 'models/improved_reinforce_final.pt'
        elif agent_type == 'mcts':
            model_path = 'models/improved_mcts_final.pt'
    
    # Create model
    if agent_type == 'dqn':
        model = DQNetwork().to(device)
    elif agent_type == 'reinforce':
        model = PolicyNetwork().to(device)
    elif agent_type == 'mcts':
        model = PolicyValueNetwork().to(device)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    # Load model weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded {agent_type} model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return 0, 0
    
    # Statistics
    wins = 0
    total_reward = 0
    gin_opportunities = 0
    knock_opportunities = 0
    gin_taken = 0
    knock_taken = 0
    
    if verbose:
        print(f"\nEvaluating {agent_type} agent over {num_games} games...")
    
    for game in range(num_games):
        state = env.reset()
        done = False
        game_reward = 0
        
        if verbose and game % 10 == 0:
            print(f"Game {game+1}/{num_games}...")
        
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
                if agent_type == 'dqn':
                    with torch.no_grad():
                        q_values = model(
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
                
                elif agent_type == 'reinforce' or agent_type == 'mcts':
                    with torch.no_grad():
                        if agent_type == 'reinforce':
                            action_probs = model(
                                state_device['hand_matrix'],
                                state_device['discard_history']
                            )
                        else:  # mcts
                            action_probs, _ = model(
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
                            # Apply mask to action probabilities
                            action_probs = action_probs.squeeze()
                            action_probs = action_probs * state_device['valid_actions_mask']
                            action_probs = action_probs / (action_probs.sum() + 1e-8)
                            action = action_probs.argmax().item()
            
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
            
            # Update state
            state = next_state
        
        # Check if player 0 (our agent) won
        if 'outcome' in info and info['outcome'] == 'win':
            wins += 1
            if verbose and game % 10 == 0:
                print(f"  Game {game+1}: Agent WON! Reward: {game_reward:.2f}")
        elif verbose and game % 10 == 0:
            print(f"  Game {game+1}: Agent lost. Reward: {game_reward:.2f}")
        
        total_reward += game_reward
    
    win_rate = wins / num_games
    avg_reward = total_reward / num_games
    
    if verbose:
        print(f"\nEvaluation Results:")
        print(f"  Win rate: {win_rate:.2f} ({wins}/{num_games})")
        print(f"  Average reward: {avg_reward:.2f}")
        
        if gin_opportunities > 0:
            print(f"  GIN opportunities: {gin_opportunities}, taken: {gin_taken} ({gin_taken/gin_opportunities*100:.1f}%)")
        if knock_opportunities > 0:
            print(f"  KNOCK opportunities: {knock_opportunities}, taken: {knock_taken} ({knock_taken/knock_opportunities*100:.1f}%)")
    
    return win_rate, avg_reward

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a Gin Rummy agent')
    parser.add_argument('--agent', type=str, default='dqn', choices=['dqn', 'reinforce', 'mcts'],
                        help='Type of agent to evaluate')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the model file')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to play')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress detailed output')
    
    args = parser.parse_args()
    
    evaluate_agent(
        agent_type=args.agent,
        model_path=args.model,
        num_games=args.games,
        verbose=not args.quiet
    ) 