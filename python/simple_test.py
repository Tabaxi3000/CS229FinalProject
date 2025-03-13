#!/usr/bin/env python3

import random
import torch
import os
import numpy as np
from simple_evaluate import (
    GinRummyGame, DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN,
    RandomPlayer, RuleBasedPlayer, GreedyPlayer
)
from dqn import DQNAgent, DQNetwork
from reinforce import REINFORCEAgent, PolicyNetwork
from mcts import MCTSAgent, PolicyValueNetwork
from gin_rummy_env import GinRummyEnv

def create_state_tensor(state):
    """Convert game state to tensor format expected by models."""
    # Create hand matrix (4x13 tensor)
    hand_matrix = torch.zeros(1, 1, 4, 13)  # [batch, channel, suit, rank]
    for card in state['playerHand']:
        suit = card // 13
        rank = card % 13
        hand_matrix[0, 0, suit, rank] = 1
    
    # Create discard history (batch_size x sequence_length x 52)
    discard_history = torch.zeros(1, 52, 52)  # [batch, seq_len, card_idx]
    for i, card in enumerate(state['discardPile']):
        discard_history[0, i, card] = 1
    
    # Create valid actions mask
    valid_actions_mask = torch.zeros(110)  # Total number of possible actions
    for action in state['validActions']:
        valid_actions_mask[action] = 1
    
    return {
        'hand_matrix': hand_matrix,
        'discard_history': discard_history,
        'valid_actions_mask': valid_actions_mask
    }

class RandomAgent:
    """A simple agent that selects random valid actions."""
    def __init__(self):
        pass
    
    def select_action(self, state):
        """Select a random valid action."""
        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
        if isinstance(valid_actions, int):
            valid_actions = [valid_actions]
        return random.choice(valid_actions)

class ModelWrapper:
    """Wrapper class to make models compatible with the evaluation framework."""
    def __init__(self, model, model_type):
        self.model = model
        self.model_type = model_type
    
    def select_action(self, state, valid_actions):
        """Select action using the wrapped model."""
        if not valid_actions:
            return None
            
        if self.model_type == 'dqn':
            state_tensor = create_state_tensor(state)
            action = self.model.select_action(state_tensor, torch.tensor(valid_actions))
            return action
            
        elif self.model_type == 'reinforce':
            state_tensor = create_state_tensor(state)
            action = self.model.select_action(state_tensor, torch.tensor(valid_actions))
            return action
            
        elif self.model_type == 'mcts':
            state_tensor = create_state_tensor(state)
            action = self.model.select_action(state_tensor)
            return action
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

def evaluate_model_against_agent(model_agent, opponent_agent, num_games=2, verbose=True):
    """Evaluate a model against another agent."""
    print(f"\nStarting evaluation: {model_agent.__class__.__name__} vs {opponent_agent.__class__.__name__} for {num_games} games")
    
    wins = 0
    losses = 0
    draws = 0
    total_score = 0
    
    for game_num in range(num_games):
        print(f"\nGame {game_num+1}/{num_games}")
        env = GinRummyEnv()
        state = env.reset()
        done = False
        turn = 0
        
        while not done and turn < 100:  # Add turn limit to prevent infinite games
            print(f"\nTurn {turn+1}")
            
            if verbose:
                env.print_state()
                
            if env.current_player == 0:  # Model's turn
                action = model_agent.select_action(state)
                print(f"Player 0 (Model) took action {action}")
            else:  # Opponent's turn
                action = opponent_agent.select_action(state)
                print(f"Player 1 (Opponent) took action {action}")
            
            state, reward, done, info = env.step(action)
            turn += 1
            
            if done and verbose:
                print(f"Game over! Reward: {reward}")
                env.print_state()
        
        # Determine winner
        if reward > 0:
            wins += 1
            print(f"Game {game_num+1}: WIN")
        elif reward < 0:
            losses += 1
            print(f"Game {game_num+1}: LOSS")
        else:
            draws += 1
            print(f"Game {game_num+1}: DRAW")
        
        total_score += reward
    
    win_rate = wins / num_games
    avg_score = total_score / num_games
    
    print(f"\nEvaluation results:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Average score: {avg_score:.2f}")
    
    return win_rate, avg_score

def main():
    """Main function to test models."""
    print("Creating models for testing...")
    
    # Create models without loading saved weights
    dqn_model = DQNAgent()
    reinforce_model = REINFORCEAgent()
    
    # Create a policy value network for MCTS
    policy_network = PolicyValueNetwork()
    value_network = PolicyValueNetwork()
    mcts_agent = MCTSAgent(policy_network, value_network, num_simulations=10)
    
    # Create a random agent for evaluation
    random_agent = RandomAgent()
    
    # Evaluate models against random agent
    print("\n--- DQN Evaluation ---")
    evaluate_model_against_agent(dqn_model, random_agent)
    
    print("\n--- REINFORCE Evaluation ---")
    evaluate_model_against_agent(reinforce_model, random_agent)
    
    print("\n--- MCTS Evaluation ---")
    evaluate_model_against_agent(mcts_agent, random_agent)
    
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 