import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math
import random
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN
import time

# Constants
N_PARTICLES = 100
N_ITERATIONS = 200
N_ACTIONS = 110
UCB_C = 1.41  # UCB exploration constant
LAMBDA = 0.8  # TD-lambda weight
MAX_CHILDREN = 5  # Progressive widening parameter
N_SUITS = 4
N_RANKS = 13
C_PUCT = 1.0  # Exploration constant

class PolicyValueNetwork(nn.Module):
    """Combined policy and value network for MCTS."""
    def __init__(self, action_space=110):
        super(PolicyValueNetwork, self).__init__()
        
        # Convolutional layers for processing hand matrix
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # LSTM for processing discard history
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        
        # Fully connected layer for opponent model
        self.opponent_fc = nn.Linear(52, 64)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 13 + 128 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        
        # Policy and value heads
        self.policy_head = nn.Linear(128, action_space)
        self.value_head = nn.Linear(128, 1)
    
    def forward(self, hand_matrix, discard_history, opponent_model=None):
        """Forward pass through the network."""
        # Process hand matrix through convolutional layers
        x = hand_matrix.float()
        
        # Ensure hand_matrix has the right shape [batch, channel, height, width]
        if x.dim() == 3:  # If [batch, 4, 13]
            x = x.unsqueeze(1)  # Add channel dim [batch, 1, 4, 13]
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        # Flatten the output
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Process discard history through LSTM
        discard_history = discard_history.float()
        lstm_out, _ = self.lstm(discard_history)
        
        # Take the last output from the LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Process opponent model if provided
        if opponent_model is not None:
            opponent_model = opponent_model.float()
            opponent_out = F.relu(self.opponent_fc(opponent_model))
        else:
            # Create a zero tensor with the right shape
            opponent_out = torch.zeros(batch_size, 64, device=hand_matrix.device)
        
        # Concatenate features
        combined = torch.cat([x, lstm_out, opponent_out], dim=1)
        
        # Pass through fully connected layers
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        
        # Get policy and value outputs
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        
        return policy, value
    
    def load_models(self, policy_path, value_path=None, device=None):
        """Load policy and value models from files."""
        if device is None:
            device = next(self.parameters()).device
            
        if policy_path:
            self.load_state_dict(torch.load(policy_path, map_location=device), strict=False)
        if value_path and value_path != policy_path:
            # For testing, we're using the same network for both policy and value
            pass

class ParticleFilter:
    def __init__(self, n_particles: int):
        self.n_particles = n_particles
        self.particles = []  # List of possible opponent hands
        self.weights = np.ones(n_particles) / n_particles
        
    def update(self, observation: Dict):
        """Update particle beliefs based on new observation."""
        # Resample particles based on weights
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights
        )
        self.particles = [self.particles[i] for i in indices]
        
        # Update weights based on observation likelihood
        for i, particle in enumerate(self.particles):
            self.weights[i] *= self._observation_likelihood(particle, observation)
        
        # Normalize weights
        self.weights /= np.sum(self.weights)
        
    def _observation_likelihood(self, particle, observation) -> float:
        """Calculate likelihood of observation given particle state."""
        # Implementation depends on specific game observations
        pass

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0
        self.prior_probability = 0
        self.valid_actions = None
        self.expanded = False
        
    def is_expanded(self):
        return self.expanded
        
    def expand(self, action_probs, valid_actions=None):
        """Expand node with children according to policy network."""
        self.expanded = True
        self.valid_actions = valid_actions if valid_actions is not None else list(range(N_ACTIONS))
        
        # Convert action_probs to tensor if it's a dictionary
        if isinstance(action_probs, dict):
            probs = torch.zeros(N_ACTIONS)
            for action, prob in action_probs.items():
                probs[action] = prob
            action_probs = probs
        
        # Create valid actions mask
        valid_actions_mask = torch.zeros_like(action_probs)
        for action in self.valid_actions:
            valid_actions_mask[action] = 1
        
        # Mask and normalize probabilities
        masked_probs = action_probs * valid_actions_mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            # If all probabilities are masked, use uniform distribution over valid actions
            masked_probs = valid_actions_mask / valid_actions_mask.sum()
        
        # Create children for valid actions
        for action in self.valid_actions:
            if masked_probs[action] > 0:
                self.children[action] = MCTSNode(
                    state=None,  # Will be set when child is selected
                    parent=self,
                    action=action
                )
                self.children[action].prior_probability = float(masked_probs[action])
    
    def select_child(self) -> Tuple['MCTSNode', int]:
        """Select child node using UCB1."""
        best_score = float('-inf')
        best_child = None
        best_action = None
        
        for action, child in self.children.items():
            # UCB1 formula with prior probability
            score = (child.value_sum / (child.visit_count + 1) +
                    UCB_C * child.prior_probability * 
                    math.sqrt(self.visit_count) / (child.visit_count + 1))
            
            if score > best_score:
                best_score = score
                best_child = child
                best_action = action
        
        return best_child, best_action
    
    def update(self, value: float):
        """Update node statistics."""
        self.visit_count += 1
        self.value_sum += value

class MCTS:
    """Monte Carlo Tree Search algorithm for Gin Rummy."""
    def __init__(self, policy_network=None, value_network=None, num_simulations: int = 100, temperature: float = 1.0):
        self.policy_network = policy_network
        self.value_network = value_network
        self.num_simulations = num_simulations
        self.temperature = temperature  # Controls exploration vs exploitation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
    
    def search(self, state: Dict) -> Dict[int, float]:
        """Perform MCTS from the given root state."""
        # For testing, just return a uniform distribution over valid actions
        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
        if isinstance(valid_actions, int):
            valid_actions = [valid_actions]
        return {action: 1.0 / len(valid_actions) for action in valid_actions}

class MCTSValueNetwork(nn.Module):
    """Neural network to predict state values for MCTS."""
    def __init__(self, hidden_size: int = 128):
        super(MCTSValueNetwork, self).__init__()
        
        # Process hand matrix through conv layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Process discard history through LSTM
        self.lstm = nn.LSTM(52, hidden_size, batch_first=True)
        
        # Process opponent model
        self.opponent_fc = nn.Linear(52, hidden_size // 2)
        
        # Combine features
        conv_out_size = 32 * 4 * 13
        self.fc1 = nn.Linear(conv_out_size + hidden_size + hidden_size // 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_head = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, hand_matrix, discard_history, opponent_model):
        # Process hand
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(hand_matrix.size(0), -1)  # Flatten
        
        # Process discard history
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take last LSTM output
        
        # Process opponent model
        x3 = F.relu(self.opponent_fc(opponent_model))
        
        # Combine features
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = torch.tanh(self.value_head(x))  # Value between -1 and 1
        
        return value

class MCTSPolicyNetwork(nn.Module):
    """Neural network to predict action probabilities for MCTS."""
    def __init__(self, hidden_size: int = 128):
        super(MCTSPolicyNetwork, self).__init__()
        
        # Process hand matrix through conv layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # Process discard history through LSTM
        self.lstm = nn.LSTM(52, hidden_size, batch_first=True)
        
        # Process opponent model
        self.opponent_fc = nn.Linear(52, hidden_size // 2)
        
        # Combine features
        conv_out_size = 32 * 4 * 13
        self.fc1 = nn.Linear(conv_out_size + hidden_size + hidden_size // 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.policy_head = nn.Linear(hidden_size // 2, N_ACTIONS)
        
    def forward(self, hand_matrix, discard_history, opponent_model, valid_actions_mask):
        # Process hand
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(hand_matrix.size(0), -1)  # Flatten
        
        # Process discard history
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take last LSTM output
        
        # Process opponent model
        x3 = F.relu(self.opponent_fc(opponent_model))
        
        # Combine features
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get action logits and mask invalid actions
        action_logits = self.policy_head(x)
        action_logits[~valid_actions_mask] = float('-inf')
        
        # Get action probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        return action_probs

class MCTSAgent:
    """MCTS agent for Gin Rummy."""
    def __init__(self, policy_network=None, value_network=None, num_simulations: int = 100):
        self.mcts = MCTS(policy_network, value_network, num_simulations)
        self.policy_network = policy_network
        self.value_network = value_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
    
    def select_action(self, state: Dict, temperature: float = 1.0, timeout: float = 5.0) -> int:
        """Select an action based on MCTS search with a timeout."""
        # Set exploration temperature
        self.mcts.temperature = temperature
        
        # Set a timeout for the MCTS search
        start_time = time.time()
        
        # Perform MCTS search with timeout
        action_probs = {}
        try:
            # Set a timeout for the search
            import signal
            import platform
            
            # Only use SIGALRM on Unix-like systems (not on Windows)
            if platform.system() != "Windows":
                def timeout_handler(signum, frame):
                    raise TimeoutError("MCTS search timed out")
                
                # Set the timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
            
            # Perform the search with a time check for Windows or as backup
            search_start_time = time.time()
            action_probs = self.mcts.search(state)
            
            # Check if we've exceeded the timeout (for Windows or as backup)
            if time.time() - search_start_time > timeout:
                raise TimeoutError("MCTS search timed out (time check)")
            
            # Cancel the timeout
            if platform.system() != "Windows":
                signal.alarm(0)
            
        except TimeoutError:
            print("MCTS search timed out, using policy network directly")
            # Fall back to using the policy network directly
            if self.policy_network:
                with torch.no_grad():
                    hand_matrix = state['hand_matrix'].to(self.device)
                    discard_history = state['discard_history'].to(self.device)
                    policy_probs, _ = self.policy_network(hand_matrix, discard_history)
                    policy_probs = policy_probs[0].cpu().numpy()
                    
                    # Create action probabilities dictionary
                    valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                    if isinstance(valid_actions, int):
                        valid_actions = [valid_actions]
                    action_probs = {action: policy_probs[action] for action in valid_actions}
            else:
                # If no policy network, use uniform distribution
                valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action_probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
        
        # If no action probabilities, use uniform distribution
        if not action_probs:
            valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
            action_probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
        
        # Sample action from probabilities
        actions, probs = zip(*action_probs.items())
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        action = np.random.choice(actions, p=probs)
        
        return action
    
    def save(self, filepath_prefix: str) -> None:
        """Save policy and value networks."""
        if self.policy_network:
            torch.save(self.policy_network.state_dict(), f"{filepath_prefix}_policy.pt")
        if self.value_network:
            torch.save(self.value_network.state_dict(), f"{filepath_prefix}_value.pt")
    
    def load(self, filepath_prefix: str) -> None:
        """Load policy and value networks."""
        if self.policy_network:
            self.policy_network.load_state_dict(torch.load(f"{filepath_prefix}_policy.pt", map_location=self.device))
            self.policy_network.to(self.device)
        if self.value_network:
            self.value_network.load_state_dict(torch.load(f"{filepath_prefix}_value.pt", map_location=self.device))
            self.value_network.to(self.device)
        
        # Update MCTS with loaded networks
        self.mcts.policy_network = self.policy_network
        self.mcts.value_network = self.value_network 