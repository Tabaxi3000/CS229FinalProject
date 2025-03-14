import os
import sys
import argparse
import time
import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

print("=== Gin Rummy Model Evaluator ===")
print("Step 1: Setting up environment and imports")

# Safely try to import PyTorch
try:
    import torch
    print("✅ PyTorch imported successfully")
except ImportError as e:
    print(f"❌ PyTorch import error: {e}")
    print("Please install PyTorch with: pip install torch")
    sys.exit(1)

# Safely try to import matplotlib for visualization
try:
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported successfully")
except ImportError as e:
    print(f"❌ Matplotlib import warning: {e}")
    print("Visualization will be disabled")

# Define model paths
MODEL_DIR = "models"
MODELS = {
    "dqn": {
        "enhanced": os.path.join(MODEL_DIR, "dqn_enhanced_best.pt"),
        "final": os.path.join(MODEL_DIR, "dqn_enhanced_final.pt"),
        "epochs": [os.path.join(MODEL_DIR, f"dqn_enhanced_epoch_{i}.pt") for i in range(1, 9)]
    },
    "reinforce": {
        "enhanced": os.path.join(MODEL_DIR, "reinforce_enhanced_best.pt"),
        "final": os.path.join(MODEL_DIR, "reinforce_enhanced_final.pt"),
        "epochs": [os.path.join(MODEL_DIR, f"reinforce_enhanced_epoch_{i}.pt") for i in range(1, 9)]
    },
    "mcts": {
        "policy": os.path.join(MODEL_DIR, "mcts_policy.pt"),
        "value": os.path.join(MODEL_DIR, "mcts_value.pt")
    }
}

# Import model classes and evaluation framework
from dqn import DQNAgent
from reinforce import REINFORCEAgent
from enhanced_train import EnhancedDQNAgent, EnhancedREINFORCEAgent
from quick_train import FastDQNAgent, FastREINFORCEAgent
from evaluate_gameplay import GinRummyGame, RuleBasedPlayer, ModelPlayer, RandomPlayer

def check_model_files():
    """Check if model files exist and return a list of available models."""
    print("\nStep 2: Checking for model files")
    available_models = {}
    
    for model_type, paths in MODELS.items():
        available_models[model_type] = {}
        for name, path in paths.items():
            if name == "epochs":
                available_epoch_models = []
                for epoch_path in path:
                    if os.path.exists(epoch_path):
                        available_epoch_models.append(epoch_path)
                if available_epoch_models:
                    print(f"✅ Found {len(available_epoch_models)} epoch models for {model_type}")
                    available_models[model_type]["epochs"] = available_epoch_models
                else:
                    print(f"❌ No epoch models found for {model_type}")
            else:
                if os.path.exists(path):
                    print(f"✅ Found {model_type} {name} model: {path}")
                    available_models[model_type][name] = path
                else:
                    print(f"❌ Missing {model_type} {name} model: {path}")
    
    return available_models

def evaluate_best_models(num_games=50):
    """Evaluate only the best models against rule-based opponents."""
    print(f"Evaluating best models with {num_games} games each...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Define the best models to evaluate
    best_models = [
        # For quick testing, let's focus on just one model at a time
        {'name': 'REINFORCE-Enhanced-Best', 'type': 'enhanced_reinforce', 'path': 'models/reinforce_enhanced_best.pt'},
        # {'name': 'DQN-Enhanced-Best', 'type': 'enhanced_dqn', 'path': 'models/dqn_enhanced_best.pt'},
        # {'name': 'REINFORCE-Quick-Final', 'type': 'fast_reinforce', 'path': 'models/reinforce_quick_final.pt'},
        # {'name': 'DQN-Quick-Final', 'type': 'fast_dqn', 'path': 'models/dqn_quick_final.pt'},
    ]
    
    # Setup opponents
    opponents = [
        {'name': 'Random', 'type': 'random'},
        {'name': 'Rule-Based', 'type': 'rule_based'}
    ]
    
    # Results storage
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Run evaluations
    for model_info in best_models:
        model_name = model_info['name']
        model_type = model_info['type']
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found, skipping...")
            continue
            
        print(f"\nEvaluating {model_name} ({model_path})...")
        
        for opponent_info in opponents:
            opponent_name = opponent_info['name']
            opponent_type = opponent_info['type']
            
            print(f"  Against {opponent_name}...")
            
            # Track statistics
            wins = 0
            losses = 0
            draws = 0
            points_scored = []
            points_allowed = []
            game_lengths = []
            
            # Play games
            for game_idx in tqdm(range(num_games), desc=f"{model_name} vs {opponent_name}"):
                # Alternate who goes first
                if game_idx % 2 == 0:
                    model_player_id = 0
                    opponent_player_id = 1
                else:
                    model_player_id = 1
                    opponent_player_id = 0
                
                # Initialize game
                game = GinRummyGame(verbose=False)
                
                # Initialize players
                model_player = ModelPlayer(model_player_id, model_type, model_path, device)
                
                if opponent_type == 'random':
                    opponent_player = RandomPlayer(opponent_player_id)
                else:  # rule_based
                    opponent_player = RuleBasedPlayer(opponent_player_id)
                
                # Play the game
                game_state = game.reset()
                done = False
                turns = 0
                max_turns = 100  # Prevent infinite games
                
                while not done and turns < max_turns:
                    # Get current player
                    current_player_id = game_state['currentPlayer']
                    
                    # Get valid actions
                    valid_actions = game.get_valid_actions(current_player_id)
                    
                    # Get player action
                    if current_player_id == model_player_id:
                        action = model_player.select_action(game_state, valid_actions)
                    else:
                        action = opponent_player.select_action(game_state, valid_actions)
                    
                    # Take action
                    game_state, reward, done = game.step(current_player_id, action)
                    turns += 1
                
                # Game over - check results
                game_lengths.append(turns)
                
                if game.winner is not None:
                    winner = game.winner
                    if winner == model_player_id:
                        wins += 1
                        # Calculate points based on deadwood difference
                        opponent_deadwood = game.calculate_deadwood(game.player_hands[opponent_player_id])
                        player_deadwood = game.calculate_deadwood(game.player_hands[model_player_id])
                        points = max(0, opponent_deadwood - player_deadwood)
                        points_scored.append(points)
                        points_allowed.append(0)
                    elif winner == opponent_player_id:
                        losses += 1
                        # Calculate points based on deadwood difference
                        opponent_deadwood = game.calculate_deadwood(game.player_hands[opponent_player_id])
                        player_deadwood = game.calculate_deadwood(game.player_hands[model_player_id])
                        points = max(0, player_deadwood - opponent_deadwood)
                        points_scored.append(0)
                        points_allowed.append(points)
                else:
                    # Draw
                    draws += 1
                    points_scored.append(0)
                    points_allowed.append(0)
            
            # Calculate stats
            win_rate = wins / num_games
            avg_points_scored = np.mean(points_scored) if points_scored else 0
            avg_points_allowed = np.mean(points_allowed) if points_allowed else 0
            avg_game_length = np.mean(game_lengths)
            
            print(f"    Results: Win Rate={win_rate:.2f}, Wins={wins}, Losses={losses}, Draws={draws}")
            print(f"    Avg Points Scored={avg_points_scored:.2f}, Avg Points Allowed={avg_points_allowed:.2f}")
            print(f"    Avg Game Length={avg_game_length:.2f} turns")
            
            # Store results
            results[model_name][opponent_name]['win_rate'] = win_rate
            results[model_name][opponent_name]['wins'] = wins
            results[model_name][opponent_name]['losses'] = losses
            results[model_name][opponent_name]['draws'] = draws
            results[model_name][opponent_name]['avg_points_scored'] = avg_points_scored
            results[model_name][opponent_name]['avg_points_allowed'] = avg_points_allowed
            results[model_name][opponent_name]['avg_game_length'] = avg_game_length
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    for model_name in results:
        print(f"\n{model_name}:")
        for opponent_name in results[model_name]:
            stats = results[model_name][opponent_name]
            print(f"  vs {opponent_name}: Win Rate={stats['win_rate']:.2f}, " + 
                  f"Wins={stats['wins']}, Losses={stats['losses']}, Draws={stats['draws']}")
    
    return results

def plot_results(results):
    """Plot evaluation results."""
    # Extract models and opponents
    models = list(results.keys())
    opponents = list(results[models[0]].keys()) if models else []
    
    if not models or not opponents:
        print("No results to plot")
        return
    
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot win rates
    win_rates = []
    for model in models:
        model_win_rates = [results[model][opponent]['win_rate'] for opponent in opponents]
        win_rates.append(model_win_rates)
    
    # Transpose for plotting
    win_rates = np.array(win_rates).T
    
    # Plot win rates bar chart
    x = np.arange(len(opponents))
    width = 0.8 / len(models)
    
    for i, model in enumerate(models):
        axes[0].bar(x + i * width - 0.4 + width/2, [results[model][opponent]['win_rate'] for opponent in opponents], 
                 width, label=model)
    
    axes[0].set_xlabel('Opponent')
    axes[0].set_ylabel('Win Rate')
    axes[0].set_title('Win Rate by Model and Opponent')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(opponents)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot average game length
    for i, model in enumerate(models):
        axes[1].bar(x + i * width - 0.4 + width/2, [results[model][opponent]['avg_game_length'] for opponent in opponents], 
                 width, label=model)
    
    axes[1].set_xlabel('Opponent')
    axes[1].set_ylabel('Average Game Length (turns)')
    axes[1].set_title('Average Game Length by Model and Opponent')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(opponents)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png')
    print("Results plot saved to model_evaluation_results.png")
    plt.show()

def main():
    # Run evaluation
    results = evaluate_best_models(num_games=10)
    
    # Plot results
    plot_results(results)

if __name__ == "__main__":
    main() 