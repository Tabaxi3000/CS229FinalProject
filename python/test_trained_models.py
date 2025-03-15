import torch
import numpy as np
import os
import random
from dqn import DQNetwork, DQNAgent
from reinforce import PolicyNetwork
from mcts import PolicyValueNetwork
from simple_evaluate import GinRummyEnv, RandomAgent
from tqdm import tqdm
from improved_gin_rummy_env import GinRummy
from rules_based_agent import RulesAgent

def create_state_tensor(state):
    """Create state tensor from state dictionary."""
    hand_matrix = state['hand_matrix'].unsqueeze(0)  # Add batch dimension
    discard_history = state['discard_history'].unsqueeze(0)  # Add batch dimension
    valid_actions_mask = state['valid_actions_mask']
    
    return {
        'hand_matrix': hand_matrix,
        'discard_history': discard_history,
        'valid_actions_mask': valid_actions_mask
    }

class ModelWrapper:
    """Wrapper for models to use in evaluation."""
    def __init__(self, model_type, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
        
        self.model_type = model_type
        
        if model_type == 'dqn':
            self.model = DQNetwork().to(self.device)
            if model_path:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        elif model_type == 'reinforce':
            self.model = PolicyNetwork().to(self.device)
            if model_path:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        elif model_type == 'mcts':
            self.model = PolicyValueNetwork().to(self.device)
            if model_path:
                policy_path = model_path.replace('_value_', '_policy_')
                value_path = model_path
                self.model.load_models(policy_path, value_path, self.device)
            self.model.eval()
        elif model_type == 'rules':
            self.model = RulesAgent()
    
    def select_action(self, state):
        """Select action based on model type."""
        with torch.no_grad():
            state_tensor = create_state_tensor(state)
            
            # Move tensors to device
            state_tensor['hand_matrix'] = state_tensor['hand_matrix'].to(self.device)
            state_tensor['discard_history'] = state_tensor['discard_history'].to(self.device)
            state_tensor['valid_actions_mask'] = state_tensor['valid_actions_mask'].to(self.device)
            
            if self.model_type == 'dqn':
                q_values = self.model(
                    state_tensor['hand_matrix'],
                    state_tensor['discard_history'],
                    state_tensor['valid_actions_mask']
                )
                
                # Mask invalid actions
                mask = state_tensor['valid_actions_mask']
                if mask.dim() == 1 and q_values.dim() > 1:
                    mask = mask.unsqueeze(0)
                    if mask.size(1) != q_values.size(1):
                        # If dimensions don't match, fall back to random valid action
                        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                        if isinstance(valid_actions, int):
                            valid_actions = [valid_actions]
                        return random.choice(valid_actions)
                    mask = mask.expand_as(q_values)
                    q_values = q_values.masked_fill(~mask.bool(), float('-inf'))
                else:
                    q_values = q_values.masked_fill(~mask.bool(), float('-inf'))
                
                return q_values.argmax().item()
            
            elif self.model_type == 'reinforce':
                action_probs = self.model(
                    state_tensor['hand_matrix'],
                    state_tensor['discard_history']
                )
                
                # Mask invalid actions
                mask = state_tensor['valid_actions_mask']
                if mask.dim() == 1 and action_probs.dim() > 1:
                    mask = mask.unsqueeze(0)
                    if mask.size(1) != action_probs.size(1):
                        # If dimensions don't match, fall back to random valid action
                        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                        if isinstance(valid_actions, int):
                            valid_actions = [valid_actions]
                        return random.choice(valid_actions)
                    mask = mask.expand_as(action_probs)
                    action_probs = action_probs.masked_fill(~mask.bool(), float('-inf'))
                else:
                    action_probs = action_probs.masked_fill(~mask.bool(), float('-inf'))
                
                return action_probs.argmax().item()
            
            elif self.model_type == 'mcts':
                # For MCTS, we use the policy network to select actions
                policy_output = self.model.policy_forward(
                    state_tensor['hand_matrix'],
                    state_tensor['discard_history']
                )
                
                # Mask invalid actions
                mask = state_tensor['valid_actions_mask']
                if mask.dim() == 1 and policy_output.dim() > 1:
                    mask = mask.unsqueeze(0)
                    if mask.size(1) != policy_output.size(1):
                        # If dimensions don't match, fall back to random valid action
                        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                        if isinstance(valid_actions, int):
                            valid_actions = [valid_actions]
                        return random.choice(valid_actions)
                    mask = mask.expand_as(policy_output)
                    policy_output = policy_output.masked_fill(~mask.bool(), float('-inf'))
                else:
                    policy_output = policy_output.masked_fill(~mask.bool(), float('-inf'))
                
                return policy_output.argmax().item()

def evaluate_model_against_agent(model, opponent, num_games=10, verbose=True):
    """Evaluate model against an opponent agent."""
    env = GinRummyEnv()
    
    wins = 0
    losses = 0
    draws = 0
    total_score = 0
    
    # Use tqdm for progress bar
    game_iterator = tqdm(range(num_games), desc=f"Evaluating {model.model_type.upper()} model") if verbose else range(num_games)
    
    for game_num in game_iterator:
        if verbose:
            print(f"\nGame {game_num + 1}/{num_games}")
        
        state = env.reset()
        done = False
        turn = 0
        
        while not done:
            turn += 1
            if verbose and turn % 10 == 0:
                print(f"Turn {turn}")
            
            # Current player's turn
            if env.current_player == 0:  # Model's turn
                action = model.select_action(state)
                if verbose:
                    print(f"Player 0 (Model) took action {action}")
            else:  # Opponent's turn
                action = opponent.select_action(state)
                if verbose:
                    print(f"Player 1 (Opponent) took action {action}")
            
            # Take action
            state, reward, done, _ = env.step(action)
            
            if verbose and turn % 10 == 0:
                print(f"=== Current Game State ===")
                print(f"Current player: {env.current_player}")
                print(f"Phase: {'discard' if env.phase == 1 else 'draw'}")
                print(f"Player 0 hand{' (current)' if env.current_player == 0 else ''}: {env.player_hands[0]}")
                print(f"Player 1 hand{' (current)' if env.current_player == 1 else ''}: {env.player_hands[1]}")
                print(f"Top discard: {env.discard_pile[-1] if env.discard_pile else 'None'}")
                print(f"Cards in deck: {len(env.deck)}")
                print(f"Current player deadwood: {env.calculate_deadwood(env.player_hands[env.current_player])}")
                print(f"Valid actions:")
                for action_idx in torch.nonzero(state['valid_actions_mask']).squeeze().tolist():
                    if isinstance(action_idx, int):
                        action_idx = [action_idx]
                    for idx in action_idx:
                        action_name = env.get_action_name(idx)
                        print(f"  {action_name} ({idx})")
                print("=========================")
        
        # Game is done, check result
        if reward > 0:
            wins += 1
            if verbose:
                print(f"Game {game_num + 1}: WIN")
        elif reward < 0:
            losses += 1
            if verbose:
                print(f"Game {game_num + 1}: LOSS")
        else:
            draws += 1
            if verbose:
                print(f"Game {game_num + 1}: DRAW")
        
        total_score += reward
        
        # Update progress bar
        if verbose:
            game_iterator.set_postfix(wins=wins, losses=losses, draws=draws, win_rate=f"{wins/(game_num+1):.2f}")
    
    # Print results
    win_rate = wins / num_games
    avg_score = total_score / num_games
    
    print(f"Evaluation results:")
    print(f"Wins: {wins}, Losses: {losses}, Draws: {draws}")
    print(f"Win rate: {win_rate:.2f}")
    print(f"Average score: {avg_score:.2f}")
    print("Evaluation complete!")
    
    return win_rate, avg_score

def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check if models directory exists
    if not os.path.exists('models'):
        print("Models directory not found. Please train models first.")
        return
    
    # Load models
    print("Loading models...")
    
    # Load DQN model
    dqn_model_path = 'models/dqn_improved.pt'
    if os.path.exists(dqn_model_path):
        dqn_model = ModelWrapper('dqn', dqn_model_path)
        print(f"DQN model loaded from {dqn_model_path}")
    else:
        dqn_model_path = 'models/dqn_enhanced_final.pt'
        if os.path.exists(dqn_model_path):
            dqn_model = ModelWrapper('dqn', dqn_model_path)
            print(f"DQN model loaded from {dqn_model_path}")
        else:
            print("DQN model not found. Skipping DQN evaluation.")
            dqn_model = None
    
    # Load REINFORCE model
    reinforce_model_path = 'models/reinforce_improved.pt'
    if os.path.exists(reinforce_model_path):
        reinforce_model = ModelWrapper('reinforce', reinforce_model_path)
        print(f"REINFORCE model loaded from {reinforce_model_path}")
    else:
        reinforce_model_path = 'models/reinforce_enhanced_final.pt'
        if os.path.exists(reinforce_model_path):
            reinforce_model = ModelWrapper('reinforce', reinforce_model_path)
            print(f"REINFORCE model loaded from {reinforce_model_path}")
        else:
            print("REINFORCE model not found. Skipping REINFORCE evaluation.")
            reinforce_model = None
    
    # Load MCTS model
    mcts_model_path = 'models/mcts_value_improved.pt'
    if os.path.exists(mcts_model_path):
        mcts_model = ModelWrapper('mcts', mcts_model_path)
        print(f"MCTS model loaded from {mcts_model_path}")
    else:
        mcts_model_path = 'models/mcts_value_final.pt'
        if os.path.exists(mcts_model_path):
            mcts_model = ModelWrapper('mcts', mcts_model_path)
            print(f"MCTS model loaded from {mcts_model_path}")
        else:
            print("MCTS model not found. Skipping MCTS evaluation.")
            mcts_model = None
    
    # Create random agent
    random_agent = RandomAgent()
    
    # Evaluate models against random agent
    print("\nEvaluating models against random agent...")
    
    if dqn_model:
        print("\nEvaluating DQN model against random agent...")
        dqn_win_rate, dqn_avg_score = evaluate_model_against_agent(dqn_model, random_agent, num_games=5)
    
    if reinforce_model:
        print("\nEvaluating REINFORCE model against random agent...")
        reinforce_win_rate, reinforce_avg_score = evaluate_model_against_agent(reinforce_model, random_agent, num_games=5)
    
    if mcts_model:
        print("\nEvaluating MCTS model against random agent...")
        mcts_win_rate, mcts_avg_score = evaluate_model_against_agent(mcts_model, random_agent, num_games=5)
    
    # Compare models
    print("\nModel comparison:")
    if dqn_model:
        print(f"DQN win rate: {dqn_win_rate:.2f}, avg score: {dqn_avg_score:.2f}")
    if reinforce_model:
        print(f"REINFORCE win rate: {reinforce_win_rate:.2f}, avg score: {reinforce_avg_score:.2f}")
    if mcts_model:
        print(f"MCTS win rate: {mcts_win_rate:.2f}, avg score: {mcts_avg_score:.2f}")

if __name__ == "__main__":
    main() 