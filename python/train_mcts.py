import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
import time
import os
import json
import numpy as np
from typing import Dict, List, Tuple
from mcts import PolicyValueNetwork, MCTSAgent, N_ACTIONS

# Constants
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
CHECKPOINT_INTERVAL = 1

def create_hand_matrix(cards: List[int]) -> np.ndarray:
    """
    Convert list of card indices to 4x13 matrix.
    
    Args:
        cards: List of card indices (0-51)
        
    Returns:
        4x13 matrix representing the hand
    """
    matrix = np.zeros((4, 13), dtype=np.float32)
    for card_idx in cards:
        suit = card_idx // 13
        rank = card_idx % 13
        matrix[suit, rank] = 1
    return matrix

def create_discard_history(discards: List[int], max_len: int = 52) -> np.ndarray:
    """
    Convert discard pile to one-hot sequence.
    
    Args:
        discards: List of discarded card indices
        max_len: Maximum length of history to consider
        
    Returns:
        Sequence of one-hot encoded cards
    """
    history = np.zeros((max_len, 52), dtype=np.float32)
    for i, card_idx in enumerate(discards[-max_len:]):
        if card_idx >= 0 and i < max_len:  # Skip invalid indices
            history[i, card_idx] = 1
    return history

def get_action_idx(action_str: str) -> int:
    """
    Convert action string to action index.
    
    Args:
        action_str: String representation of action
        
    Returns:
        Action index (0-109)
    """
    if action_str.startswith('draw_faceup_'):
        return 0 if action_str.endswith('True') else 1
    elif action_str.startswith('discard_'):
        try:
            card_id = int(action_str.split('_')[1])
            return 2 + card_id
        except:
            return -1
    elif action_str == 'knock':
        return 108
    elif action_str == 'gin':
        return 109
    else:
        return -1

def load_game_data(file_path: str) -> List[Dict]:
    """
    Load game states from a JSON file.
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        List of game state dictionaries
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to list if it's a dictionary
    if isinstance(data, dict):
        states = data.get('gameStates', [])
    else:
        states = data
    
    return states

def prepare_batch(states: List[Dict], device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of states for training.
    
    Args:
        states: List of game state dictionaries
        device: Device to move tensors to
        
    Returns:
        Tuple of (state_batch, action_batch, reward_batch)
    """
    hand_matrices = []
    discard_histories = []
    actions = []
    rewards = []
    
    for state in states:
        # Create hand matrix
        hand_matrix = create_hand_matrix(state['playerHand'])
        hand_matrices.append(hand_matrix)
        
        # Create discard history
        discard_history = create_discard_history(state['discardPile'])
        discard_histories.append(discard_history)
        
        # Get action index
        action_idx = get_action_idx(state['action'])
        if action_idx >= 0:
            action = torch.zeros(N_ACTIONS)
            action[action_idx] = 1.0
            actions.append(action)
        else:
            action = torch.zeros(N_ACTIONS)
            actions.append(action)
        
        # Get reward
        reward = state['reward']
        rewards.append(reward)
    
    # Convert to tensors
    batch = {
        'hand_matrix': torch.FloatTensor(np.array(hand_matrices))[:, None, :, :].to(device),  # Add channel dim
        'discard_history': torch.FloatTensor(np.array(discard_histories)).to(device)
    }
    
    return (
        batch,
        torch.stack(actions).to(device),
        torch.FloatTensor(rewards).to(device)
    )

def train_mcts(data_files: List[str], num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE, num_simulations: int = 100):
    """
    Train an MCTS agent for Gin Rummy as described in the CS229 milestone.
    
    This function trains a combined policy-value network using supervised learning
    on game data, which will be used by the MCTS algorithm for action selection.
    
    Args:
        data_files: List of paths to game data files
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        num_simulations: Number of MCTS simulations to use for evaluation
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize network
    policy_value_net = PolicyValueNetwork().to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(policy_value_net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Loss functions
    policy_criterion = nn.CrossEntropyLoss()
    value_criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        total_batches = 0
        
        # Shuffle files for each epoch
        random.shuffle(data_files)
        
        # Process each file
        for file_idx, file_path in enumerate(data_files):
            print(f"\nFile {file_idx + 1}/{len(data_files)}: {file_path}")
            
            # Load data
            states = load_game_data(file_path)
            print(f"Loaded {len(states)} states")
            
            # Process in batches
            num_batches = len(states) // batch_size
            indices = list(range(len(states)))
            random.shuffle(indices)
            
            progress_bar = tqdm(range(min(num_batches, 500)), desc="Training")  # Limit to 500 batches per file
            
            for batch_idx in progress_bar:
                # Select a batch of states
                batch_indices = indices[batch_idx * batch_size:(batch_idx + 1) * batch_size]
                batch_states = [states[i] for i in batch_indices]
                
                # Prepare batch
                state_batch, action_batch, reward_batch = prepare_batch(batch_states, device)
                
                # Forward pass
                optimizer.zero_grad()
                policy_output, value_output = policy_value_net(
                    state_batch['hand_matrix'],
                    state_batch['discard_history']
                )
                
                # Compute policy loss
                policy_loss = policy_criterion(policy_output, action_batch)
                
                # Compute value loss
                value_output = value_output.squeeze(1)
                value_loss = value_criterion(value_output, reward_batch)
                
                # Compute total loss
                loss = policy_loss + value_loss
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(policy_value_net.parameters(), 5.0)
                
                # Update weights
                optimizer.step()
                
                # Track losses
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                total_batches += 1
                
                # Update progress bar
                progress_bar.set_postfix({
                    'p_loss': policy_loss.item(),
                    'v_loss': value_loss.item()
                })
        
        # Calculate average losses
        avg_policy_loss = epoch_policy_loss / max(1, total_batches)
        avg_value_loss = epoch_value_loss / max(1, total_batches)
        print(f"Epoch {epoch + 1} average losses - Policy: {avg_policy_loss:.6f}, Value: {avg_value_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            torch.save(policy_value_net.state_dict(), f"models/mcts_policy_value_epoch_{epoch + 1}.pt")
            print(f"Saved checkpoint for epoch {epoch + 1}")
    
    # Save final model
    torch.save(policy_value_net.state_dict(), "models/mcts_policy_value_final.pt")
    print("Saved final model")
    
    # Create and save agent
    agent = MCTSAgent(policy_value_net, num_simulations=num_simulations)
    agent.save("models/mcts")
    print("Saved MCTS agent")

def evaluate_mcts(test_file: str, num_simulations: int = 100):
    """
    Evaluate a trained MCTS agent.
    
    Args:
        test_file: Path to test data file
        num_simulations: Number of MCTS simulations to use
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    # Load network
    policy_value_net = PolicyValueNetwork().to(device)
    
    try:
        policy_value_net.load_state_dict(torch.load("models/mcts_policy_value_final.pt", map_location=device))
        print("Loaded trained model")
    except FileNotFoundError:
        print("No trained model found, using untrained network")
    
    # Create MCTS agent
    agent = MCTSAgent(policy_value_net, num_simulations=num_simulations)
    
    # Load test data
    states = load_game_data(test_file)
    print(f"Loaded {len(states)} test states")
    
    # Evaluate on test data
    correct = 0
    total = 0
    
    for state in tqdm(states, desc="Evaluating"):
        # Convert state to format expected by agent
        agent_state = {
            'hand': state['playerHand'],
            'discard_pile': state['discardPile'],
            'stock_pile': state.get('stockPile', []),
            'can_draw': True
        }
        
        # Get agent's action
        agent_action = agent.select_action(agent_state)
        
        # Get ground truth action
        true_action = get_action_idx(state['action'])
        
        # Check if correct
        if agent_action == true_action:
            correct += 1
        total += 1
    
    # Calculate accuracy
    accuracy = correct / total if total > 0 else 0
    print(f"Evaluation accuracy: {accuracy:.4f} ({correct}/{total})")

def main():
    """Main function to train and evaluate MCTS agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train and evaluate MCTS agent for Gin Rummy')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing game data files')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--simulations', type=int, default=100, help='Number of MCTS simulations')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate, do not train')
    parser.add_argument('--test-file', type=str, help='File to use for evaluation')
    
    args = parser.parse_args()
    
    # Find training data files
    data_files = []
    if os.path.exists(args.data_dir):
        for file in os.listdir(args.data_dir):
            if file.endswith('.json'):
                data_files.append(os.path.join(args.data_dir, file))
    
    if not data_files:
        print(f"No data files found in {args.data_dir}")
        return
    
    print(f"Found {len(data_files)} data files")
    
    # Train or evaluate
    if not args.eval_only:
        train_mcts(data_files, args.epochs, args.batch_size, args.simulations)
    
    # Evaluate
    if args.test_file:
        evaluate_mcts(args.test_file, args.simulations)
    elif not args.eval_only:
        # Use last file for evaluation
        evaluate_mcts(data_files[-1], args.simulations)

if __name__ == "__main__":
    main() 