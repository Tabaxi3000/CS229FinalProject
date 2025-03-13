#!/usr/bin/env python3

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from improved_gin_rummy_env import ImprovedGinRummyEnv, DRAW_STOCK, DRAW_DISCARD, DISCARD_START, DISCARD_END, KNOCK, GIN
from improved_quick_train import DQNetwork, PolicyNetwork, PolicyValueNetwork

def analyze_q_values(model, state, device, env):
    """Analyze Q-values for a given state."""
    # Move state to device
    state_device = {
        'hand_matrix': state['hand_matrix'].to(device),
        'discard_history': state['discard_history'].to(device),
        'valid_actions_mask': state['valid_actions_mask'].to(device)
    }
    
    # Get Q-values
    with torch.no_grad():
        q_values = model(
            state_device['hand_matrix'],
            state_device['discard_history']
        ).squeeze()
    
    # Get valid actions
    valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
    if isinstance(valid_actions, int):
        valid_actions = [valid_actions]
    
    # Print Q-values for valid actions
    print("\nQ-values for valid actions:")
    for action in valid_actions:
        action_name = get_action_name(action, env)
        print(f"  {action_name}: {q_values[action].item():.4f}")
    
    # Find best action
    masked_q_values = q_values.clone()
    masked_q_values[~state_device['valid_actions_mask'].bool()] = float('-inf')
    best_action = masked_q_values.argmax().item()
    
    print(f"\nBest action: {get_action_name(best_action, env)} (Q-value: {q_values[best_action].item():.4f})")
    
    return best_action, q_values

def get_action_name(action, env):
    """Get human-readable name for an action."""
    if action == DRAW_STOCK:
        return "DRAW_STOCK"
    elif action == DRAW_DISCARD:
        return "DRAW_DISCARD"
    elif action == KNOCK:
        return "KNOCK"
    elif action == GIN:
        return "GIN"
    elif DISCARD_START <= action <= DISCARD_END:
        card_idx = action - DISCARD_START
        card_str = env._format_cards([card_idx])[0]
        return f"DISCARD {card_str}"
    else:
        return f"Unknown action {action}"

def analyze_hand(hand, env):
    """Analyze a hand for melds and deadwood."""
    melds = env._find_melds(hand)
    deadwood = env._calculate_deadwood(hand)
    
    print("\nHand analysis:")
    print(f"  Cards: {env._format_cards(hand)}")
    
    if melds:
        print("  Melds:")
        for i, meld in enumerate(melds):
            print(f"    Meld {i+1}: {env._format_cards(meld)}")
    else:
        print("  No melds found")
    
    print(f"  Deadwood count: {deadwood}")
    
    # Calculate deadwood cards
    all_meld_cards = []
    for meld in melds:
        all_meld_cards.extend(meld)
    
    deadwood_cards = [card for card in hand if card not in all_meld_cards]
    if deadwood_cards:
        print(f"  Deadwood cards: {env._format_cards(deadwood_cards)}")
    
    return melds, deadwood

def analyze_game_step(model, state, env, device, step_num):
    """Analyze a single step in the game."""
    print(f"\n===== Step {step_num} =====")
    
    # Print current player
    print(f"Current player: {'Agent' if env.current_player == 0 else 'Opponent'}")
    
    # Analyze hand
    hand = env.player_hands[0] if env.current_player == 0 else env.player_hands[1]
    melds, deadwood = analyze_hand(hand, env)
    
    # Print discard pile
    if env.discard_pile:
        print(f"\nDiscard pile top: {env._format_cards([env.discard_pile[-1]])[0]}")
    else:
        print("\nDiscard pile is empty")
    
    # If it's the agent's turn, analyze Q-values
    if env.current_player == 0:
        best_action, q_values = analyze_q_values(model, state, device, env)
        return best_action
    else:
        return None

def visualize_q_values(q_values, valid_actions, env, step_num):
    """Visualize Q-values for valid actions."""
    # Filter for valid actions
    valid_q_values = []
    action_names = []
    
    for action in valid_actions:
        valid_q_values.append(q_values[action].item())
        action_names.append(get_action_name(action, env))
    
    # Create bar chart
    plt.figure(figsize=(10, 6))
    bars = plt.bar(action_names, valid_q_values)
    
    # Color the highest value bar
    max_idx = np.argmax(valid_q_values)
    bars[max_idx].set_color('red')
    
    plt.title(f'Q-values for Valid Actions (Step {step_num})')
    plt.xlabel('Action')
    plt.ylabel('Q-value')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(f'q_values_step_{step_num}.png')
    plt.close()

def analyze_agent_decisions(model_path, num_games=5, max_steps=100, visualize=False):
    """Analyze agent's decisions over multiple games."""
    # Set up environment
    env = ImprovedGinRummyEnv(reward_shaping=True)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    # Load model
    model = DQNetwork().to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Loaded DQN model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Statistics
    games_won = 0
    total_rewards = 0
    gin_opportunities = 0
    gin_taken = 0
    knock_opportunities = 0
    knock_taken = 0
    
    for game in range(num_games):
        print(f"\n\n========== Game {game+1} ==========")
        state = env.reset()
        done = False
        step = 0
        game_reward = 0
        
        while not done and step < max_steps:
            # Get valid actions
            valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
            
            # If it's the agent's turn
            if env.current_player == 0:
                # Track GIN and KNOCK opportunities
                if GIN in valid_actions:
                    gin_opportunities += 1
                if KNOCK in valid_actions:
                    knock_opportunities += 1
                
                # Analyze this step
                best_action = analyze_game_step(model, state, env, device, step)
                
                # Track if GIN or KNOCK was taken
                if best_action == GIN:
                    gin_taken += 1
                elif best_action == KNOCK:
                    knock_taken += 1
                
                # Visualize Q-values if requested
                if visualize:
                    with torch.no_grad():
                        q_values = model(
                            state['hand_matrix'].to(device),
                            state['discard_history'].to(device)
                        ).squeeze()
                    visualize_q_values(q_values, valid_actions, env, step)
                
                # Take action
                next_state, reward, done, _, info = env.step(best_action)
                game_reward += reward
                
                print(f"Action taken: {get_action_name(best_action, env)}")
                print(f"Reward: {reward:.4f}")
                
                if done:
                    print("\nGame over!")
                    if 'outcome' in info:
                        print(f"Outcome: {info['outcome']}")
                        if info['outcome'] == 'win' or info['outcome'] == 'gin':
                            games_won += 1
                            print("Agent won!")
                        else:
                            print("Agent lost!")
                    print(f"Final reward: {game_reward:.4f}")
                    break
            else:
                # Opponent's turn - random action
                action = np.random.choice(valid_actions)
                next_state, reward, done, _, info = env.step(action)
                game_reward += reward
                
                print(f"\n===== Step {step} =====")
                print("Opponent's turn")
                print(f"Action taken: {get_action_name(action, env)}")
                
                if done:
                    print("\nGame over!")
                    if 'outcome' in info:
                        print(f"Outcome: {info['outcome']}")
                        if info['outcome'] == 'win' or info['outcome'] == 'gin':
                            print("Agent lost!")
                        else:
                            games_won += 1
                            print("Agent won!")
                    print(f"Final reward: {game_reward:.4f}")
                    break
            
            state = next_state
            step += 1
        
        total_rewards += game_reward
        print(f"Game {game+1} completed in {step} steps with reward {game_reward:.4f}")
    
    # Print overall statistics
    print("\n===== Overall Statistics =====")
    print(f"Games played: {num_games}")
    print(f"Games won: {games_won} ({games_won/num_games*100:.1f}%)")
    print(f"Average reward: {total_rewards/num_games:.4f}")
    
    if gin_opportunities > 0:
        print(f"GIN opportunities: {gin_opportunities}")
        print(f"GIN taken: {gin_taken} ({gin_taken/gin_opportunities*100:.1f}%)")
    
    if knock_opportunities > 0:
        print(f"KNOCK opportunities: {knock_opportunities}")
        print(f"KNOCK taken: {knock_taken} ({knock_taken/knock_opportunities*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze DQN agent decisions')
    parser.add_argument('--model', type=str, default="models/improved_dqn_final.pt",
                        help='Path to DQN model')
    parser.add_argument('--games', type=int, default=5,
                        help='Number of games to analyze')
    parser.add_argument('--max-steps', type=int, default=100,
                        help='Maximum steps per game')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize Q-values (saves PNG files)')
    
    args = parser.parse_args()
    analyze_agent_decisions(args.model, args.games, args.max_steps, args.visualize) 