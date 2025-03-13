#!/usr/bin/env python3

import torch
import random
import numpy as np
import argparse
import os
from tqdm import tqdm
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN
from improved_quick_train import DQNetwork, PolicyNetwork, PolicyValueNetwork, MCTSAgent

def get_agent_action(agent_type, model, state, device, valid_actions):
    """Get action from agent based on state."""
    # Move state to device
    state_device = {
        'hand_matrix': state['hand_matrix'].to(device),
        'discard_history': state['discard_history'].to(device),
        'valid_actions_mask': state['valid_actions_mask'].to(device)
    }
    
    # Prioritize GIN and KNOCK actions
    if GIN in valid_actions:
        return GIN
    elif KNOCK in valid_actions:
        return KNOCK
    
    # Get action based on agent type
    with torch.no_grad():
        if agent_type == 'dqn':
            q_values = model(
                state_device['hand_matrix'],
                state_device['discard_history']
            )
            # Apply mask to Q-values
            masked_q_values = q_values.squeeze().clone()
            masked_q_values[~state_device['valid_actions_mask'].bool()] = float('-inf')
            action = masked_q_values.argmax().item()
            
        elif agent_type == 'reinforce':
            # Handle both single output and tuple output models
            output = model(
                state_device['hand_matrix'],
                state_device['discard_history']
            )
            
            # Check if output is a tuple (action_probs, value) or just action_probs
            if isinstance(output, tuple):
                action_probs = output[0]
            else:
                action_probs = output
                
            # Apply mask to action probabilities
            action_probs = action_probs.squeeze()
            action_probs = action_probs * state_device['valid_actions_mask']
            action_probs = action_probs / (action_probs.sum() + 1e-8)
            action = action_probs.argmax().item()
            
        elif agent_type == 'mcts':
            if isinstance(model, MCTSAgent):
                # If model is an MCTS agent, use its select_action method
                action = model.select_action(state, valid_actions, training=False)
            else:
                # If model is a PolicyValueNetwork, use it to get action probabilities
                action_probs, _ = model(
                    state_device['hand_matrix'],
                    state_device['discard_history']
                )
                # Apply mask to action probabilities
                action_probs = action_probs.squeeze()
                action_probs = action_probs * state_device['valid_actions_mask']
                action_probs = action_probs / (action_probs.sum() + 1e-8)
                action = action_probs.argmax().item()
                
        elif agent_type == 'random':
            action = random.choice(valid_actions)
            
    return action

def play_game(agent1_type, agent1_model, agent2_type, agent2_model, device, env, verbose=False):
    """Play a game between two agents and return the winner."""
    state = env.reset()
    done = False
    
    while not done:
        # Get valid actions
        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
        if isinstance(valid_actions, int):
            valid_actions = [valid_actions]
        
        # Determine which agent's turn it is
        if env.current_player == 0:  # Agent 1's turn
            action = get_agent_action(agent1_type, agent1_model, state, device, valid_actions)
        else:  # Agent 2's turn
            action = get_agent_action(agent2_type, agent2_model, state, device, valid_actions)
        
        # Take action
        next_state, reward, done, _, info = env.step(action)
        
        # Print action if verbose
        if verbose:
            player = "Agent 1" if env.current_player == 1 else "Agent 2"
            if action == GIN:
                print(f"{player} chose GIN")
            elif action == KNOCK:
                print(f"{player} chose KNOCK")
            elif action == DRAW_STOCK:
                print(f"{player} drew from stock")
            elif action == DRAW_DISCARD:
                print(f"{player} drew from discard")
            elif DISCARD_START <= action <= DISCARD_END:
                card_idx = action - DISCARD_START
                card_str = env._format_cards([card_idx])[0]
                print(f"{player} discarded {card_str}")
        
        # Update state
        state = next_state
    
    # Determine winner
    if 'outcome' in info:
        if info['outcome'] == 'win' or info['outcome'] == 'gin':
            winner = 0 if env.current_player == 1 else 1
        else:
            winner = 1 if env.current_player == 1 else 0
    else:
        # Fallback if outcome not in info
        winner = 0 if reward > 0 else 1
    
    if verbose:
        print(f"Game over! {'Agent 1' if winner == 0 else 'Agent 2'} wins!")
        print(f"Final deadwood - Agent 1: {info.get('player_deadwood', 'N/A')}, Agent 2: {info.get('opponent_deadwood', 'N/A')}")
    
    return winner

def load_model_with_fallbacks(model_type, model_path, device):
    """Load model with fallbacks for different state dict formats."""
    if model_type == 'dqn':
        model = DQNetwork().to(device)
        try:
            # Try direct loading
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded dqn model from {model_path}")
            return model
        except Exception as e:
            # Try loading with 'policy_state_dict' key
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'policy_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['policy_state_dict'])
                    print(f"Loaded dqn model from {model_path} using policy_state_dict key")
                    return model
            except:
                pass
            
            print(f"Error loading dqn model: {e}")
            return None
            
    elif model_type == 'reinforce':
        model = PolicyNetwork().to(device)
        try:
            # Try direct loading
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Loaded reinforce model from {model_path}")
            return model
        except Exception as e:
            # Try loading with 'policy_state_dict' key
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'policy_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['policy_state_dict'])
                    print(f"Loaded reinforce model from {model_path} using policy_state_dict key")
                    return model
                elif 'policy' in checkpoint:
                    model.load_state_dict(checkpoint['policy'])
                    print(f"Loaded reinforce model from {model_path} using policy key")
                    return model
            except Exception as inner_e:
                print(f"Error in fallback loading for reinforce: {inner_e}")
                pass
            
            print(f"Error loading reinforce model: {e}")
            return None
            
    elif model_type == 'mcts':
        try:
            # Try loading as PolicyValueNetwork
            policy_value_net = PolicyValueNetwork().to(device)
            try:
                policy_value_net.load_state_dict(torch.load(model_path, map_location=device))
                print(f"Loaded mcts policy-value network from {model_path}")
                return MCTSAgent(policy_value_net, device, num_simulations=50)
            except Exception as e:
                # Try loading with 'policy_state_dict' key
                try:
                    checkpoint = torch.load(model_path, map_location=device)
                    if 'policy_state_dict' in checkpoint:
                        policy_value_net.load_state_dict(checkpoint['policy_state_dict'])
                        print(f"Loaded mcts policy-value network from {model_path} using policy_state_dict key")
                        return MCTSAgent(policy_value_net, device, num_simulations=50)
                except:
                    pass
                
                print(f"Error loading mcts model: {e}")
                return None
        except Exception as e:
            print(f"Error creating MCTS agent: {e}")
            return None
    
    return None

def compare_agents(agent1_type, agent1_path, agent2_type, agent2_path, num_games=100, verbose=False):
    """Compare two agents by playing multiple games."""
    # Set up environment
    env = ImprovedGinRummyEnv(reward_shaping=False)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    # Load agent 1 model
    agent1_model = None
    if agent1_type != 'random':
        agent1_model = load_model_with_fallbacks(agent1_type, agent1_path, device)
        if agent1_model is None:
            print(f"Failed to load agent 1 model. Aborting comparison.")
            return None
        agent1_model.eval()
    
    # Load agent 2 model
    agent2_model = None
    if agent2_type != 'random':
        agent2_model = load_model_with_fallbacks(agent2_type, agent2_path, device)
        if agent2_model is None:
            print(f"Failed to load agent 2 model. Aborting comparison.")
            return None
        agent2_model.eval()
    
    # Play games
    agent1_wins = 0
    agent2_wins = 0
    
    print(f"\nComparing {agent1_type.upper()} vs {agent2_type.upper()} over {num_games} games...")
    
    for game in tqdm(range(num_games), desc="Playing games"):
        winner = play_game(agent1_type, agent1_model, agent2_type, agent2_model, device, env, verbose=(verbose and game < 5))
        if winner == 0:
            agent1_wins += 1
        else:
            agent2_wins += 1
    
    # Print results
    print(f"\nResults after {num_games} games:")
    print(f"  {agent1_type.upper()}: {agent1_wins} wins ({agent1_wins/num_games*100:.1f}%)")
    print(f"  {agent2_type.upper()}: {agent2_wins} wins ({agent2_wins/num_games*100:.1f}%)")
    
    return agent1_wins, agent2_wins

def run_tournament(model_paths, num_games=50, verbose=False):
    """Run a tournament between all agent types."""
    agent_types = ['dqn', 'reinforce', 'mcts', 'random']
    results = {}
    
    # Initialize results dictionary
    for agent1 in agent_types:
        results[agent1] = {}
        for agent2 in agent_types:
            if agent1 != agent2:
                results[agent1][agent2] = 0
    
    # Run all matchups
    for i, agent1 in enumerate(agent_types):
        for agent2 in agent_types[i+1:]:  # Only play each matchup once
            agent1_path = model_paths.get(agent1)
            agent2_path = model_paths.get(agent2)
            
            comparison_result = compare_agents(agent1, agent1_path, agent2, agent2_path, num_games, verbose)
            if comparison_result is None:
                print(f"Skipping {agent1} vs {agent2} due to model loading errors")
                continue
                
            agent1_wins, agent2_wins = comparison_result
            
            # Store results
            results[agent1][agent2] = agent1_wins / num_games
            results[agent2][agent1] = agent2_wins / num_games
    
    # Print tournament results
    print("\n===== TOURNAMENT RESULTS =====")
    print("Win rates:")
    
    # Calculate total win rate for each agent
    total_wins = {agent: 0 for agent in agent_types}
    total_games = {agent: 0 for agent in agent_types}
    
    for agent1 in agent_types:
        for agent2 in agent_types:
            if agent1 != agent2 and agent2 in results.get(agent1, {}):
                win_rate = results[agent1][agent2]
                total_wins[agent1] += win_rate * num_games
                total_games[agent1] += num_games
    
    # Print results in order of performance
    sorted_agents = sorted(agent_types, key=lambda x: total_wins[x] / total_games[x] if total_games[x] > 0 else 0, reverse=True)
    
    for i, agent in enumerate(sorted_agents):
        if total_games[agent] > 0:
            win_rate = total_wins[agent] / total_games[agent]
            print(f"{i+1}. {agent.upper()}: {win_rate*100:.1f}% win rate")
    
    print("\nDetailed matchups:")
    print("       ", end="")
    for agent in agent_types:
        print(f"{agent.ljust(10)}", end="")
    print()
    
    for agent1 in agent_types:
        print(f"{agent1.ljust(7)}", end="")
        for agent2 in agent_types:
            if agent1 == agent2:
                print("---      ", end="")
            elif agent2 in results.get(agent1, {}):
                win_rate = results[agent1][agent2]
                print(f"{win_rate*100:.1f}%    ", end="")
            else:
                print("N/A      ", end="")
        print()
    
    print("============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare Gin Rummy agents')
    parser.add_argument('--dqn', type=str, default="models/improved_dqn_final.pt",
                        help='Path to DQN model')
    parser.add_argument('--reinforce', type=str, default="models/improved_reinforce_final.pt",
                        help='Path to REINFORCE model')
    parser.add_argument('--mcts', type=str, default="models/improved_mcts_final.pt",
                        help='Path to MCTS model')
    parser.add_argument('--games', type=int, default=50,
                        help='Number of games to play for each matchup')
    parser.add_argument('--verbose', action='store_true',
                        help='Print detailed game information')
    parser.add_argument('--tournament', action='store_true',
                        help='Run a tournament between all agent types')
    parser.add_argument('--agent1', type=str, default='dqn',
                        choices=['dqn', 'reinforce', 'mcts', 'random'],
                        help='First agent type')
    parser.add_argument('--agent2', type=str, default='random',
                        choices=['dqn', 'reinforce', 'mcts', 'random'],
                        help='Second agent type')
    
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Set up model paths
    model_paths = {
        'dqn': args.dqn,
        'reinforce': args.reinforce,
        'mcts': args.mcts,
        'random': None
    }
    
    if args.tournament:
        run_tournament(model_paths, args.games, args.verbose)
    else:
        compare_agents(args.agent1, model_paths[args.agent1], 
                      args.agent2, model_paths[args.agent2], 
                      args.games, args.verbose) 