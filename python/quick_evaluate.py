import torch
import numpy as np
import random
import time
import os
import json
from typing import Dict, List

# Import model classes
try:
    from quick_train import FastDQNAgent, FastREINFORCEAgent
    from enhanced_train import EnhancedDQNAgent, EnhancedREINFORCEAgent
except ImportError as e:
    print(f"Error importing model classes: {e}")

# Constants
TEST_SAMPLES = 10  # Number of state samples to test

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

def load_state_samples(file_path: str, num_samples: int = TEST_SAMPLES) -> List[Dict]:
    """Load a few random states from a file for testing."""
    print(f"Loading samples from {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert to list if it's a dictionary
        if isinstance(data, dict):
            states = data.get('gameStates', [])
        else:
            states = data
        
        # Get random samples
        if len(states) > num_samples:
            samples = random.sample(states, num_samples)
        else:
            samples = states
            
        print(f"Loaded {len(samples)} state samples for testing")
        return samples
    except Exception as e:
        print(f"Error loading data: {e}")
        return []

def load_dqn_agent(model_path: str, agent_type: str = "fast"):
    """Load a DQN agent with its weights."""
    try:
        if agent_type == "fast":
            agent = FastDQNAgent()
            # Add load method
            checkpoint = torch.load(model_path)
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.steps_done = checkpoint.get('steps_done', 0)
        else:  # enhanced
            agent = EnhancedDQNAgent()
            # Add load method
            checkpoint = torch.load(model_path)
            agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            agent.target_net.load_state_dict(checkpoint['target_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.steps_done = checkpoint.get('steps_done', 0)
        
        return agent
    except Exception as e:
        print(f"Error loading DQN model from {model_path}: {e}")
        return None

def load_reinforce_agent(model_path: str, agent_type: str = "fast"):
    """Load a REINFORCE agent with its weights."""
    try:
        if agent_type == "fast":
            agent = FastREINFORCEAgent()
            # Add load method
            checkpoint = torch.load(model_path)
            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:  # enhanced
            agent = EnhancedREINFORCEAgent()
            # Add load method
            checkpoint = torch.load(model_path)
            agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return agent
    except Exception as e:
        print(f"Error loading REINFORCE model from {model_path}: {e}")
        return None

def prepare_state_for_dqn(state: Dict, device: torch.device) -> Dict:
    """Convert state dict to format expected by DQN models."""
    # Create hand matrix
    hand_matrix = torch.FloatTensor(create_hand_matrix(state['playerHand'])).unsqueeze(0).unsqueeze(0)
    
    # Create discard history
    discard_history = torch.FloatTensor(create_discard_history(state['discardPile'])).unsqueeze(0)
    
    # Create valid actions mask - for simplicity, allow all actions
    valid_actions_mask = torch.ones(1, 110, dtype=torch.bool)
    
    # Move tensors to device
    hand_matrix = hand_matrix.to(device)
    discard_history = discard_history.to(device)
    valid_actions_mask = valid_actions_mask.to(device)
    
    return {
        'hand_matrix': hand_matrix,
        'discard_history': discard_history,
        'valid_actions_mask': valid_actions_mask
    }

def prepare_state_for_reinforce(state: Dict, device: torch.device) -> Dict:
    """Convert state dict to format expected by REINFORCE models."""
    # Create hand matrix
    hand_matrix = torch.FloatTensor(create_hand_matrix(state['playerHand'])).unsqueeze(0).unsqueeze(0)
    
    # Create discard history
    discard_history = torch.FloatTensor(create_discard_history(state['discardPile'])).unsqueeze(0)
    
    # Create opponent model
    opponent_model = torch.FloatTensor(create_hand_matrix(state.get('knownOpponentCards', []))).reshape(1, -1)
    
    # Create valid actions mask - for simplicity, allow all actions
    valid_actions_mask = torch.ones(1, 110, dtype=torch.bool)
    
    # Move tensors to device
    hand_matrix = hand_matrix.to(device)
    discard_history = discard_history.to(device)
    opponent_model = opponent_model.to(device)
    valid_actions_mask = valid_actions_mask.to(device)
    
    return {
        'hand_matrix': hand_matrix,
        'discard_history': discard_history,
        'opponent_model': opponent_model,
        'valid_actions_mask': valid_actions_mask
    }

def test_dqn_model(agent, state_samples: List[Dict], device: torch.device, name: str):
    """Test a DQN model's inference time on sample states."""
    print(f"\nTesting {name}...")
    
    agent.policy_net.eval()  # Set to evaluation mode
    inference_times = []
    
    for state in state_samples:
        # Prepare state
        dqn_state = prepare_state_for_dqn(state, device)
        
        # Measure inference time
        start_time = time.time()
        
        # Get model prediction
        with torch.no_grad():
            q_values = agent.policy_net(
                dqn_state['hand_matrix'],
                dqn_state['discard_history'],
                dqn_state['valid_actions_mask']
            )
            action = q_values.argmax(dim=1).item()
        
        # Record inference time
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
    
    # Calculate average inference time
    avg_inference_time = sum(inference_times) / len(inference_times) * 1000  # ms
    print(f"  Average inference time: {avg_inference_time:.2f} ms")
    
    return avg_inference_time

def test_reinforce_model(agent, state_samples: List[Dict], device: torch.device, name: str):
    """Test a REINFORCE model's inference time on sample states."""
    print(f"\nTesting {name}...")
    
    agent.policy.eval()  # Set to evaluation mode
    inference_times = []
    
    for state in state_samples:
        # Prepare state
        reinforce_state = prepare_state_for_reinforce(state, device)
        
        # Measure inference time
        start_time = time.time()
        
        # Get model prediction
        with torch.no_grad():
            action_probs = agent.policy(
                reinforce_state['hand_matrix'],
                reinforce_state['discard_history'],
                reinforce_state['opponent_model'],
                reinforce_state['valid_actions_mask']
            )
            action = action_probs.argmax(dim=1).item()
        
        # Record inference time
        inference_time = time.time() - start_time
        inference_times.append(inference_time)
    
    # Calculate average inference time
    avg_inference_time = sum(inference_times) / len(inference_times) * 1000  # ms
    print(f"  Average inference time: {avg_inference_time:.2f} ms")
    
    return avg_inference_time

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Load sample states
    test_file = "../java/MavenProject/training_data_consolidated_10.json"
    
    if not os.path.exists(test_file):
        print(f"Test file {test_file} not found!")
        for i in range(10, 0, -1):
            alt_file = f"../java/MavenProject/training_data_consolidated_{i}.json"
            if os.path.exists(alt_file):
                test_file = alt_file
                print(f"Using {test_file} as test file instead.")
                break
        else:
            print("No test files found!")
            return
    
    state_samples = load_state_samples(test_file)
    if not state_samples:
        print("No state samples found for testing!")
        return
    
    # Models to test
    models = [
        # Quick models
        {"type": "dqn", "agent_type": "fast", "name": "DQN-Quick", "path": "models/dqn_quick_final.pt"},
        {"type": "reinforce", "agent_type": "fast", "name": "REINFORCE-Quick", "path": "models/reinforce_quick_final.pt"},
        
        # Enhanced models
        {"type": "dqn", "agent_type": "enhanced", "name": "DQN-Enhanced", "path": "models/dqn_enhanced_final.pt"},
        {"type": "reinforce", "agent_type": "enhanced", "name": "REINFORCE-Enhanced", "path": "models/reinforce_enhanced_final.pt"},
    ]
    
    # Results
    results = {}
    
    # Test each model
    for model_info in models:
        try:
            model_type = model_info["type"]
            agent_type = model_info["agent_type"]
            model_name = model_info["name"]
            model_path = model_info["path"]
            
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found, skipping...")
                continue
            
            if model_type == "dqn":
                agent = load_dqn_agent(model_path, agent_type)
                if agent:
                    avg_time = test_dqn_model(agent, state_samples, device, model_name)
                    results[model_name] = avg_time
            elif model_type == "reinforce":
                agent = load_reinforce_agent(model_path, agent_type)
                if agent:
                    avg_time = test_reinforce_model(agent, state_samples, device, model_name)
                    results[model_name] = avg_time
        except Exception as e:
            print(f"Error testing {model_info['name']}: {e}")
    
    # Print summary
    if results:
        print("\n=== PERFORMANCE SUMMARY ===")
        for model_name, avg_time in sorted(results.items(), key=lambda x: x[1]):
            print(f"{model_name}: {avg_time:.2f} ms")
        
        # Find fastest model
        fastest_model = min(results.items(), key=lambda x: x[1])
        print(f"\nFastest model: {fastest_model[0]} ({fastest_model[1]:.2f} ms)")
    else:
        print("\nNo models were successfully tested!")

if __name__ == "__main__":
    main() 