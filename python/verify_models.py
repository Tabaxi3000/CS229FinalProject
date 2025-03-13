import os
import torch
import numpy as np
import time

# Try importing model classes
from dqn import DQNAgent
from reinforce import REINFORCEAgent
from enhanced_train import EnhancedDQNAgent, EnhancedREINFORCEAgent
from quick_train import FastDQNAgent, FastREINFORCEAgent

# Define utility functions for state preparation
def create_hand_matrix(cards):
    """Create a 4x13 matrix representation of a hand."""
    # Initialize an empty matrix (4 suits x 13 ranks)
    matrix = np.zeros((4, 13), dtype=np.float32)
    
    # For each card, set the corresponding position to 1
    for card in cards:
        suit = card // 13
        rank = card % 13
        matrix[suit, rank] = 1.0
    
    return matrix

def create_discard_history(discards, max_len=52):
    """Create a sequence of one-hot vectors for discard history."""
    # Initialize an empty history (all zeros)
    history = np.zeros((max_len, 52), dtype=np.float32)
    
    # For each discard, create a one-hot vector
    for i, card in enumerate(discards[-max_len:]):
        history[i, card] = 1.0
    
    return history

def create_dummy_state():
    """Create a dummy game state for testing model predictions."""
    # Create a random hand (10 cards)
    hand = np.random.choice(52, 10, replace=False).tolist()
    
    # Create a random discard pile (5 cards)
    discards = np.random.choice([i for i in range(52) if i not in hand], 5, replace=False).tolist()
    
    # Create a state dictionary similar to what would be used in the game
    state = {
        'playerHand': hand,
        'discardPile': discards,
        'knownOpponentCards': [],
        'drawnCard': None,
        'currentPlayer': 0,
        'turnCount': 3,
        'gameOver': False,
        'winner': None,
        'validActions': list(range(110))  # All actions are valid for testing
    }
    
    return state

def prepare_state_for_model(state, device):
    """Prepare a state for model input."""
    # Create hand matrix
    hand_matrix = torch.FloatTensor(create_hand_matrix(state['playerHand'])).unsqueeze(0).unsqueeze(0)
    
    # Create discard history
    discard_history = torch.FloatTensor(create_discard_history(state['discardPile'])).unsqueeze(0)
    
    # Create valid actions mask
    valid_actions_mask = torch.ones(1, 110, dtype=torch.bool)
    
    # Create opponent model (dummy for REINFORCE, zeros for DQN)
    opponent_model = torch.zeros(1, 52)
    
    # Move tensors to device
    hand_matrix = hand_matrix.to(device)
    discard_history = discard_history.to(device)
    valid_actions_mask = valid_actions_mask.to(device)
    opponent_model = opponent_model.to(device)
    
    return {
        'hand_matrix': hand_matrix,
        'discard_history': discard_history,
        'valid_actions_mask': valid_actions_mask,
        'opponent_model': opponent_model
    }

def load_model(model_path, model_type):
    """Load a model from a checkpoint file."""
    try:
        if model_type == 'dqn':
            agent = DQNAgent()
            checkpoint = torch.load(model_path)
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return agent
        elif model_type == 'reinforce':
            agent = REINFORCEAgent()
            checkpoint = torch.load(model_path)
            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return agent
        elif model_type == 'enhanced_dqn':
            agent = EnhancedDQNAgent()
            checkpoint = torch.load(model_path)
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return agent
        elif model_type == 'enhanced_reinforce':
            agent = EnhancedREINFORCEAgent()
            checkpoint = torch.load(model_path)
            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return agent
        elif model_type == 'fast_dqn':
            agent = FastDQNAgent()
            checkpoint = torch.load(model_path)
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return agent
        elif model_type == 'fast_reinforce':
            agent = FastREINFORCEAgent()
            checkpoint = torch.load(model_path)
            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            return agent
        else:
            print(f"Unknown model type: {model_type}")
            return None
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

def test_model_prediction(agent, model_type, prepared_state):
    """Test model prediction on a prepared state."""
    start_time = time.time()
    
    try:
        if model_type in ['dqn', 'enhanced_dqn', 'fast_dqn']:
            # DQN models predict Q-values for each action
            with torch.no_grad():
                q_values = agent.policy_net(
                    prepared_state['hand_matrix'],
                    prepared_state['discard_history'],
                    prepared_state['valid_actions_mask']
                )
                # Get predicted action (highest Q-value)
                predicted_action = q_values.argmax(dim=1).item()
        else:  # REINFORCE models
            # REINFORCE models predict action probabilities
            with torch.no_grad():
                action_probs = agent.policy(
                    prepared_state['hand_matrix'],
                    prepared_state['discard_history'],
                    prepared_state['opponent_model'],
                    prepared_state['valid_actions_mask']
                )
                # Get predicted action (highest probability)
                predicted_action = action_probs.argmax(dim=1).item()
        
        elapsed_time = (time.time() - start_time) * 1000  # ms
        return predicted_action, elapsed_time
    except Exception as e:
        print(f"Error in model prediction: {e}")
        return None, 0

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Models to test
    models = [
        {'name': 'REINFORCE-Enhanced-Best', 'type': 'enhanced_reinforce', 'path': 'models/reinforce_enhanced_best.pt'},
        {'name': 'DQN-Enhanced-Best', 'type': 'enhanced_dqn', 'path': 'models/dqn_enhanced_best.pt'},
        {'name': 'REINFORCE-Quick-Final', 'type': 'fast_reinforce', 'path': 'models/reinforce_quick_final.pt'},
        {'name': 'DQN-Quick-Final', 'type': 'fast_dqn', 'path': 'models/dqn_quick_final.pt'},
    ]
    
    # Create dummy state
    state = create_dummy_state()
    prepared_state = prepare_state_for_model(state, device)
    
    print("\n=== TESTING MODEL LOADING AND PREDICTION ===")
    
    for model_info in models:
        model_name = model_info['name']
        model_type = model_info['type']
        model_path = model_info['path']
        
        if not os.path.exists(model_path):
            print(f"Model file {model_path} not found, skipping...")
            continue
            
        print(f"\nTesting {model_name} ({model_path})...")
        
        # Load model
        agent = load_model(model_path, model_type)
        if agent is None:
            print("  ❌ Model loading failed")
            continue
        
        print("  ✅ Model loaded successfully")
        
        # Test prediction
        predicted_action, elapsed_time = test_model_prediction(agent, model_type, prepared_state)
        
        if predicted_action is not None:
            print(f"  ✅ Prediction successful: Action={predicted_action}, Time={elapsed_time:.2f}ms")
        else:
            print("  ❌ Prediction failed")
    
    print("\n=== MODEL VERIFICATION COMPLETE ===")

if __name__ == "__main__":
    main() 