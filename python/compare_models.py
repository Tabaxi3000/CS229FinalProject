import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import random
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# Import model architectures
from dqn import DQN as DQNAgent 
from reinforce import REINFORCEAgent
from mcts import MCTSAgent, MCTSPolicyNetwork, MCTSValueNetwork

# Import model architectures from quick_train
from quick_train import SimpleDQNetwork, SimplePolicyNetwork, SimpleReplayBuffer, SimpleDQNAgent, SimpleREINFORCEAgent

# Import model architectures from enhanced_train
from enhanced_train import EnhancedDQNetwork, EnhancedPolicyNetwork, EnhancedDQNAgent, EnhancedREINFORCEAgent

# Constants
TEST_BATCH_SIZE = 100
MAX_TEST_STATES = 1000

def load_game_data(file_path: str) -> List[Dict]:
    """Load game states from a JSON file."""
    print(f"Loading test data from {file_path}")
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Convert to list if it's a dictionary
    if isinstance(data, dict):
        states = data.get('gameStates', [])
    else:
        states = data
    
    print(f"Loaded {len(states)} states")
    return states

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

class ModelEvaluator:
    """Class to evaluate different models on the same test data."""
    def __init__(self, test_file: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
        print(f"Using device: {self.device}")
        
        # Load test data
        self.test_states = load_game_data(test_file)
        # Sample subset of states for efficiency
        if len(self.test_states) > MAX_TEST_STATES:
            self.test_states = random.sample(self.test_states, MAX_TEST_STATES)
        
        # Results storage
        self.results = {}
    
    def prepare_state_for_dqn(self, state: Dict) -> Dict:
        """Convert state dict to format expected by DQN models."""
        # Create hand matrix
        hand_matrix = torch.FloatTensor(create_hand_matrix(state['playerHand'])).unsqueeze(0).unsqueeze(0)
        
        # Create discard history
        discard_history = torch.FloatTensor(create_discard_history(state['discardPile'])).unsqueeze(0)
        
        # Create valid actions mask - for simplicity, allow all actions
        valid_actions_mask = torch.ones(1, 110, dtype=torch.bool)
        
        # Move tensors to device
        hand_matrix = hand_matrix.to(self.device)
        discard_history = discard_history.to(self.device)
        valid_actions_mask = valid_actions_mask.to(self.device)
        
        return {
            'hand_matrix': hand_matrix,
            'discard_history': discard_history,
            'valid_actions_mask': valid_actions_mask
        }
    
    def prepare_state_for_reinforce(self, state: Dict) -> Dict:
        """Convert state dict to format expected by REINFORCE models."""
        # Create hand matrix
        hand_matrix = torch.FloatTensor(create_hand_matrix(state['playerHand'])).unsqueeze(0).unsqueeze(0)
        
        # Create discard history
        discard_history = torch.FloatTensor(create_discard_history(state['discardPile'])).unsqueeze(0)
        
        # Create opponent model
        opponent_model = torch.FloatTensor(create_hand_matrix(state['knownOpponentCards'])).reshape(1, -1)
        
        # Create valid actions mask - for simplicity, allow all actions
        valid_actions_mask = torch.ones(1, 110, dtype=torch.bool)
        
        # Move tensors to device
        hand_matrix = hand_matrix.to(self.device)
        discard_history = discard_history.to(self.device)
        opponent_model = opponent_model.to(self.device)
        valid_actions_mask = valid_actions_mask.to(self.device)
        
        return {
            'hand_matrix': hand_matrix,
            'discard_history': discard_history,
            'opponent_model': opponent_model,
            'valid_actions_mask': valid_actions_mask
        }
    
    def prepare_state_for_mcts(self, state: Dict) -> Dict:
        """Convert state dict to format expected by MCTS."""
        return {
            'playerHand': state['playerHand'],
            'discardPile': state['discardPile'],
            'knownOpponentCards': state['knownOpponentCards'],
            'faceUpCard': state.get('faceUpCard', -1),
            'phase': 'draw' if state['action'].startswith('draw') else 'discard',
            'isTerminal': state['isTerminal'],
            'reward': state['reward']
        }
    
    def evaluate_dqn(self, model_name: str, agent, num_samples: int = TEST_BATCH_SIZE):
        """Evaluate a DQN model on test data."""
        print(f"Evaluating {model_name}...")
        agent.policy_net.eval()  # Set to evaluation mode
        
        correct_actions = 0
        total_actions = 0
        inference_times = []
        
        # Sample states for evaluation
        eval_states = random.sample(self.test_states, min(num_samples, len(self.test_states)))
        
        for state in tqdm(eval_states, desc=f"Evaluating {model_name}"):
            # Skip terminal states and invalid actions
            if state['isTerminal'] or get_action_idx(state['action']) < 0:
                continue
                
            # Prepare state
            dqn_state = self.prepare_state_for_dqn(state)
            
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
            
            # Check accuracy
            true_action = get_action_idx(state['action'])
            if action == true_action:
                correct_actions += 1
            total_actions += 1
        
        # Calculate metrics
        accuracy = correct_actions / max(1, total_actions) * 100
        avg_inference_time = sum(inference_times) / max(1, len(inference_times)) * 1000  # ms
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'correct_actions': correct_actions,
            'total_actions': total_actions
        }
        
        print(f"{model_name} - Accuracy: {accuracy:.2f}% ({correct_actions}/{total_actions})")
        print(f"{model_name} - Avg inference time: {avg_inference_time:.2f} ms")
        
        return accuracy, avg_inference_time
    
    def evaluate_reinforce(self, model_name: str, agent, num_samples: int = TEST_BATCH_SIZE):
        """Evaluate a REINFORCE model on test data."""
        print(f"Evaluating {model_name}...")
        agent.policy.eval()  # Set to evaluation mode
        
        correct_actions = 0
        total_actions = 0
        inference_times = []
        
        # Sample states for evaluation
        eval_states = random.sample(self.test_states, min(num_samples, len(self.test_states)))
        
        for state in tqdm(eval_states, desc=f"Evaluating {model_name}"):
            # Skip terminal states and invalid actions
            if state['isTerminal'] or get_action_idx(state['action']) < 0:
                continue
                
            # Prepare state
            reinforce_state = self.prepare_state_for_reinforce(state)
            
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
            
            # Check accuracy
            true_action = get_action_idx(state['action'])
            if action == true_action:
                correct_actions += 1
            total_actions += 1
        
        # Calculate metrics
        accuracy = correct_actions / max(1, total_actions) * 100
        avg_inference_time = sum(inference_times) / max(1, len(inference_times)) * 1000  # ms
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'correct_actions': correct_actions,
            'total_actions': total_actions
        }
        
        print(f"{model_name} - Accuracy: {accuracy:.2f}% ({correct_actions}/{total_actions})")
        print(f"{model_name} - Avg inference time: {avg_inference_time:.2f} ms")
        
        return accuracy, avg_inference_time
    
    def evaluate_mcts(self, model_name: str, agent, num_samples: int = TEST_BATCH_SIZE, num_simulations: int = 10):
        """Evaluate an MCTS model on test data."""
        print(f"Evaluating {model_name}...")
        
        correct_actions = 0
        total_actions = 0
        inference_times = []
        
        # Sample states for evaluation - use fewer samples since MCTS is slower
        eval_states = random.sample(self.test_states, min(num_samples // 10, len(self.test_states)))
        
        for state in tqdm(eval_states, desc=f"Evaluating {model_name}"):
            # Skip terminal states and invalid actions
            if state['isTerminal'] or get_action_idx(state['action']) < 0:
                continue
                
            # Prepare state
            mcts_state = self.prepare_state_for_mcts(state)
            
            # Measure inference time
            start_time = time.time()
            
            # Get model prediction with low temperature for deterministic play
            action = agent.select_action(mcts_state, temperature=0.1)
            
            # Record inference time
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Check accuracy
            true_action = get_action_idx(state['action'])
            if action == true_action:
                correct_actions += 1
            total_actions += 1
        
        # Calculate metrics
        accuracy = correct_actions / max(1, total_actions) * 100
        avg_inference_time = sum(inference_times) / max(1, len(inference_times)) * 1000  # ms
        
        # Store results
        self.results[model_name] = {
            'accuracy': accuracy,
            'avg_inference_time': avg_inference_time,
            'correct_actions': correct_actions,
            'total_actions': total_actions
        }
        
        print(f"{model_name} - Accuracy: {accuracy:.2f}% ({correct_actions}/{total_actions})")
        print(f"{model_name} - Avg inference time: {avg_inference_time:.2f} ms")
        
        return accuracy, avg_inference_time
    
    def visualize_results(self):
        """Visualize the comparison results."""
        if not self.results:
            print("No results to visualize.")
            return
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Extract data
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        inference_times = [self.results[name]['avg_inference_time'] for name in model_names]
        
        # Plot accuracy
        ax1.bar(model_names, accuracies, color='skyblue')
        ax1.set_title('Model Accuracy')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_ylim(0, 100)
        
        # Add accuracy values on bars
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 1, f"{v:.1f}%", ha='center')
        
        # Plot inference time
        ax2.bar(model_names, inference_times, color='salmon')
        ax2.set_title('Model Inference Time')
        ax2.set_ylabel('Average Inference Time (ms)')
        
        # Add inference time values on bars
        for i, v in enumerate(inference_times):
            ax2.text(i, v + 1, f"{v:.1f} ms", ha='center')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        print("Saved visualization to model_comparison.png")
        
        # Generate text summary
        print("\n=== MODEL COMPARISON SUMMARY ===")
        
        # Sort models by accuracy
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (name, metrics) in enumerate(sorted_models):
            print(f"{i+1}. {name}:")
            print(f"   Accuracy: {metrics['accuracy']:.2f}% ({metrics['correct_actions']}/{metrics['total_actions']})")
            print(f"   Avg. Inference Time: {metrics['avg_inference_time']:.2f} ms")
            
        print("\nBest model by accuracy: " + sorted_models[0][0])
        
        # Sort by inference time
        fastest_model = min(self.results.items(), key=lambda x: x[1]['avg_inference_time'])
        print(f"Fastest model: {fastest_model[0]} ({fastest_model[1]['avg_inference_time']:.2f} ms)")

def load_or_create_agent(model_type, model_name, model_path):
    """Load model if it exists, otherwise create a new one."""
    try:
        if model_type == "dqn":
            agent = SimpleDQNAgent() if "quick" in model_name else EnhancedDQNAgent() if "enhanced" in model_name else DQNAgent()
            # Load model
            print(f"Loading {model_path}...")
            agent.load(model_path)
            return agent
        elif model_type == "reinforce":
            agent = SimpleREINFORCEAgent() if "quick" in model_name else EnhancedREINFORCEAgent() if "enhanced" in model_name else REINFORCEAgent()
            # Load model
            print(f"Loading {model_path}...")
            agent.load(model_path)
            return agent
        elif model_type == "mcts":
            policy_network = MCTSPolicyNetwork()
            value_network = MCTSValueNetwork()
            agent = MCTSAgent(policy_network, value_network, num_simulations=10)
            # Load model
            agent.load("models/mcts")
            return agent
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        print(f"Model {model_name} not found or failed to load. Skipping.")
        return None
    
    return None

def main():
    # Find a test file
    test_file = "../java/MavenProject/training_data_consolidated_10.json"  # Use the last file as test
    
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
    
    # Create evaluator
    evaluator = ModelEvaluator(test_file)
    
    # Models to evaluate
    models = [
        # Quick models
        {"type": "dqn", "name": "DQN-Quick", "path": "models/dqn_quick_final.pt"},
        {"type": "reinforce", "name": "REINFORCE-Quick", "path": "models/reinforce_quick_final.pt"},
        
        # Enhanced models (might not be available yet)
        {"type": "dqn", "name": "DQN-Enhanced", "path": "models/dqn_enhanced_final.pt"},
        {"type": "reinforce", "name": "REINFORCE-Enhanced", "path": "models/reinforce_enhanced_final.pt"},
        
        # MCTS models (might not be available yet)
        {"type": "mcts", "name": "MCTS", "path": "models/mcts"}
    ]
    
    # Evaluate each model
    for model_info in models:
        agent = load_or_create_agent(model_info["type"], model_info["name"], model_info["path"])
        
        if agent is None:
            continue
            
        if model_info["type"] == "dqn":
            evaluator.evaluate_dqn(model_info["name"], agent)
        elif model_info["type"] == "reinforce":
            evaluator.evaluate_reinforce(model_info["name"], agent)
        elif model_info["type"] == "mcts":
            evaluator.evaluate_mcts(model_info["name"], agent, num_samples=50)  # Fewer samples for MCTS as it's slower
    
    # Visualize results
    evaluator.visualize_results()

if __name__ == "__main__":
    main() 