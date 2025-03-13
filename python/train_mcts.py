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
from mcts import MCTSPolicyNetwork, MCTSValueNetwork, MCTSAgent, N_ACTIONS

# Constants
BATCH_SIZE = 512
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 10
CHECKPOINT_INTERVAL = 1

def create_hand_matrix(cards: List[int]) -> np.ndarray:
    """Convert list of card indices to 4x13 matrix."""
    matrix = np.zeros((4, 13), dtype=np.float32)
    for card_idx in cards:
        suit = card_idx // 13
        rank = card_idx % 13
        matrix[suit, rank] = 1
    return matrix

def create_discard_history(discards: List[int], max_len: int = 52) -> np.ndarray:
    """Convert discard pile to one-hot sequence."""
    history = np.zeros((max_len, 52), dtype=np.float32)
    for i, card_idx in enumerate(discards[-max_len:]):
        if card_idx >= 0 and i < max_len:  # Skip invalid indices
            history[i, card_idx] = 1
    return history

def get_action_idx(action_str: str) -> int:
    """Convert action string to action index."""
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
    """Load game states from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to list if it's a dictionary
    if isinstance(data, dict):
        states = data.get('gameStates', [])
    else:
        states = data
    
    return states

def prepare_batch(states: List[Dict], device) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Prepare a batch of states for training."""
    hand_matrices = []
    discard_histories = []
    opponent_cards = []
    actions = []
    rewards = []
    
    for state in states:
        # Create hand matrix
        hand_matrix = create_hand_matrix(state['playerHand'])
        hand_matrices.append(hand_matrix)
        
        # Create discard history
        discard_history = create_discard_history(state['discardPile'])
        discard_histories.append(discard_history)
        
        # Create opponent known cards
        opponent_matrix = create_hand_matrix(state['knownOpponentCards'])
        opponent_cards.append(opponent_matrix.flatten())
        
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
        'discard_history': torch.FloatTensor(np.array(discard_histories)).to(device),
        'opponent_model': torch.FloatTensor(np.array(opponent_cards)).to(device),
        'valid_actions_mask': torch.ones(len(states), N_ACTIONS, dtype=torch.bool).to(device)  # All actions valid for training
    }
    
    return (
        batch,
        torch.stack(actions).to(device),
        torch.FloatTensor(rewards).to(device)
    )

def train_mcts(data_files: List[str], num_epochs: int = NUM_EPOCHS, batch_size: int = BATCH_SIZE):
    """Train an MCTS agent for Gin Rummy."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize networks
    policy_network = MCTSPolicyNetwork().to(device)
    value_network = MCTSValueNetwork().to(device)
    
    # Initialize optimizers
    policy_optimizer = optim.Adam(policy_network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    value_optimizer = optim.Adam(value_network.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
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
                
                # Train policy network
                policy_optimizer.zero_grad()
                policy_output = policy_network(
                    state_batch['hand_matrix'],
                    state_batch['discard_history'],
                    state_batch['opponent_model'],
                    state_batch['valid_actions_mask']
                )
                
                # Compute policy loss
                policy_loss = policy_criterion(policy_output, action_batch)
                policy_loss.backward()
                policy_optimizer.step()
                
                # Train value network
                value_optimizer.zero_grad()
                value_output = value_network(
                    state_batch['hand_matrix'],
                    state_batch['discard_history'],
                    state_batch['opponent_model']
                ).squeeze(1)
                
                # Compute value loss
                value_loss = value_criterion(value_output, reward_batch)
                value_loss.backward()
                value_optimizer.step()
                
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
            torch.save(policy_network.state_dict(), f"models/mcts_policy_epoch_{epoch + 1}.pt")
            torch.save(value_network.state_dict(), f"models/mcts_value_epoch_{epoch + 1}.pt")
            print(f"Saved checkpoint for epoch {epoch + 1}")
    
    # Save final model
    torch.save(policy_network.state_dict(), "models/mcts_policy_final.pt")
    torch.save(value_network.state_dict(), "models/mcts_value_final.pt")
    print("Saved final model")

def evaluate_mcts(test_file: str, num_simulations: int = 100):
    """Evaluate a trained MCTS agent."""
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    
    # Load networks
    policy_network = MCTSPolicyNetwork().to(device)
    value_network = MCTSValueNetwork().to(device)
    
    try:
        policy_network.load_state_dict(torch.load("models/mcts_policy_final.pt", map_location=device))
        value_network.load_state_dict(torch.load("models/mcts_value_final.pt", map_location=device))
        print("Loaded trained models")
    except FileNotFoundError:
        print("No trained models found, using untrained networks")
    
    # Create MCTS agent
    agent = MCTSAgent(policy_network, value_network, num_simulations)
    
    # Load test data
    states = load_game_data(test_file)
    print(f"Loaded {len(states)} test states")
    
    # Select a subset for evaluation
    eval_states = random.sample(states, min(100, len(states)))
    
    # Evaluate
    correct_actions = 0
    total_actions = 0
    
    for state in tqdm(eval_states, desc="Evaluating"):
        # Convert state to dictionary format expected by MCTS
        mcts_state = {
            'playerHand': state['playerHand'],
            'discardPile': state['discardPile'],
            'knownOpponentCards': state['knownOpponentCards'],
            'faceUpCard': state.get('faceUpCard', -1),
            'phase': 'draw' if state['action'].startswith('draw') else 'discard',
            'isTerminal': state['isTerminal'],
            'reward': state['reward']
        }
        
        # Get MCTS action
        action = agent.select_action(mcts_state, temperature=0.1)  # Low temperature for more deterministic play
        
        # Check if action matches ground truth
        true_action = get_action_idx(state['action'])
        if action == true_action:
            correct_actions += 1
        total_actions += 1
    
    # Calculate accuracy
    accuracy = correct_actions / max(1, total_actions) * 100
    print(f"Accuracy: {accuracy:.2f}% ({correct_actions}/{total_actions})")

def main():
    # Find training data files
    data_files = []
    for i in range(1, 11):
        file_path = f"../java/MavenProject/training_data_consolidated_{i}.json"
        if os.path.exists(file_path):
            data_files.append(file_path)
    
    if not data_files:
        print("No training data files found!")
        return
    
    print(f"Found {len(data_files)} training files")
    
    # Set aside the last file for evaluation
    test_file = data_files.pop()
    
    # Train
    start_time = time.time()
    print("Starting MCTS training...")
    train_mcts(data_files, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)
    
    train_time = (time.time() - start_time) / 60
    print(f"Training completed in {train_time:.2f} minutes")
    
    # Evaluate
    print("\nEvaluating trained model...")
    evaluate_mcts(test_file)

if __name__ == "__main__":
    main() 