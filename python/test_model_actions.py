import os
import torch
import numpy as np
from collections import Counter

# Import model classes
from dqn import DQNAgent
from reinforce import REINFORCEAgent
from enhanced_train import EnhancedDQNAgent, EnhancedREINFORCEAgent
from quick_train import FastDQNAgent, FastREINFORCEAgent
from verify_models import create_hand_matrix, create_discard_history

# Action constants
DRAW_STOCK = 0
DRAW_DISCARD = 1
DISCARD = 2  # Base action for discarding, add card index to get specific discard action
KNOCK = 3
GIN = 4

def create_test_states(num_states=100):
    """Create test states with varying deadwood counts."""
    states = []
    
    for _ in range(num_states):
        # Create a random hand (10 cards)
        hand = np.random.choice(52, 10, replace=False).tolist()
        
        # Create a random discard pile (5 cards)
        discards = np.random.choice([i for i in range(52) if i not in hand], 5, replace=False).tolist()
        
        # Create a drawn card (for testing discard, knock, and gin actions)
        drawn_card = np.random.choice([i for i in range(52) if i not in hand and i not in discards])
        
        # Create a state dictionary
        state = {
            'playerHand': hand,
            'discardPile': discards,
            'knownOpponentCards': [],
            'drawnCard': drawn_card,
            'currentPlayer': 0,
            'turnCount': np.random.randint(1, 20),
            'gameOver': False,
            'winner': None,
            'validActions': [DRAW_STOCK, DRAW_DISCARD, KNOCK, GIN] + [DISCARD + i for i in range(52)]
        }
        
        states.append(state)
    
    return states

def prepare_state_for_model(state, device):
    """Prepare a state for model input."""
    # Create hand matrix - include drawn card if present
    hand = state['playerHand'].copy()
    if state['drawnCard'] is not None:
        hand.append(state['drawnCard'])
    hand_matrix = torch.FloatTensor(create_hand_matrix(hand)).unsqueeze(0).unsqueeze(0)
    
    # Create discard history
    discard_history = torch.FloatTensor(create_discard_history(state['discardPile'])).unsqueeze(0)
    
    # Create valid actions mask
    valid_actions_mask = torch.zeros(1, 110, dtype=torch.bool)
    for action in state['validActions']:
        valid_actions_mask[0, action] = True
    
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

def get_model_predictions(agent, model_type, states, device):
    """Get model predictions for a set of states."""
    actions = []
    
    for state in states:
        prepared_state = prepare_state_for_model(state, device)
        
        try:
            with torch.no_grad():
                if model_type in ['dqn', 'enhanced_dqn', 'fast_dqn']:
                    # DQN models predict Q-values for each action
                    q_values = agent.policy_net(
                        prepared_state['hand_matrix'],
                        prepared_state['discard_history'],
                        prepared_state['valid_actions_mask']
                    )
                    # Get predicted action (highest Q-value)
                    action = q_values.argmax(dim=1).item()
                else:  # REINFORCE models
                    # REINFORCE models predict action probabilities
                    action_probs = agent.policy(
                        prepared_state['hand_matrix'],
                        prepared_state['discard_history'],
                        prepared_state['opponent_model'],
                        prepared_state['valid_actions_mask']
                    )
                    # Get predicted action (highest probability)
                    action = action_probs.argmax(dim=1).item()
                
                actions.append(action)
        except Exception as e:
            print(f"Error in model prediction: {e}")
            actions.append(None)
    
    return actions

def analyze_actions(actions):
    """Analyze the distribution of predicted actions."""
    # Count actions by type
    action_counts = Counter(actions)
    
    # Categorize actions
    draw_stock_count = action_counts.get(DRAW_STOCK, 0)
    draw_discard_count = action_counts.get(DRAW_DISCARD, 0)
    knock_count = action_counts.get(KNOCK, 0)
    gin_count = action_counts.get(GIN, 0)
    
    # Count discard actions
    discard_count = sum(action_counts.get(DISCARD + i, 0) for i in range(52))
    
    # Total valid predictions
    total_valid = draw_stock_count + draw_discard_count + knock_count + gin_count + discard_count
    
    # Calculate percentages
    if total_valid > 0:
        draw_stock_pct = draw_stock_count / total_valid * 100
        draw_discard_pct = draw_discard_count / total_valid * 100
        knock_pct = knock_count / total_valid * 100
        gin_pct = gin_count / total_valid * 100
        discard_pct = discard_count / total_valid * 100
    else:
        draw_stock_pct = draw_discard_pct = knock_pct = gin_pct = discard_pct = 0
    
    results = {
        'draw_stock': {'count': draw_stock_count, 'percentage': draw_stock_pct},
        'draw_discard': {'count': draw_discard_count, 'percentage': draw_discard_pct},
        'knock': {'count': knock_count, 'percentage': knock_pct},
        'gin': {'count': gin_count, 'percentage': gin_pct},
        'discard': {'count': discard_count, 'percentage': discard_pct},
        'total_valid': total_valid,
        'total_none': actions.count(None)
    }
    
    return results

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
    
    # Create test states
    print("Creating test states...")
    states = create_test_states(num_states=100)
    print(f"Created {len(states)} test states")
    
    # Test each model
    print("\n=== TESTING MODEL ACTION DISTRIBUTION ===")
    
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
        
        # Get model predictions
        print("  Getting model predictions...")
        actions = get_model_predictions(agent, model_type, states, device)
        
        # Analyze actions
        results = analyze_actions(actions)
        
        # Print results
        print(f"  Action Distribution (Total: {results['total_valid']} valid, {results['total_none']} failed):")
        print(f"    Draw Stock: {results['draw_stock']['count']} ({results['draw_stock']['percentage']:.1f}%)")
        print(f"    Draw Discard: {results['draw_discard']['count']} ({results['draw_discard']['percentage']:.1f}%)")
        print(f"    Knock: {results['knock']['count']} ({results['knock']['percentage']:.1f}%)")
        print(f"    Gin: {results['gin']['count']} ({results['gin']['percentage']:.1f}%)")
        print(f"    Discard: {results['discard']['count']} ({results['discard']['percentage']:.1f}%)")
    
    print("\n=== ACTION TESTING COMPLETE ===")

if __name__ == "__main__":
    main() 