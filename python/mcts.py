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
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        self.opponent_fc = nn.Linear(52, 64)
        self.fc1 = nn.Linear(64 * 4 * 13 + 128 + 64, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_head = nn.Linear(128, action_space)
        self.value_head = nn.Linear(128, 1)
    def forward(self, hand_matrix, discard_history, opponent_model=None):
        """Forward pass through the network."""
        x = hand_matrix.float()
        if x.dim() == 3:  # If [batch, 4, 13]
            x = x.unsqueeze(1)  # Add channel dim [batch, 1, 4, 13]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        discard_history = discard_history.float()
        lstm_out, _ = self.lstm(discard_history)
        lstm_out = lstm_out[:, -1, :]
        if opponent_model is not None:
            opponent_model = opponent_model.float()
            opponent_out = F.relu(self.opponent_fc(opponent_model))
        else:
            opponent_out = torch.zeros(batch_size, 64, device=hand_matrix.device)
        combined = torch.cat([x, lstm_out, opponent_out], dim=1)
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
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
            pass
class ParticleFilter:
    def __init__(self, n_particles: int):
        self.n_particles = n_particles
        self.particles = []  # List of possible opponent hands
        self.weights = np.ones(n_particles) / n_particles
    def update(self, observation: Dict):
        """Update particle beliefs based on new observation."""
        indices = np.random.choice(
            self.n_particles,
            size=self.n_particles,
            p=self.weights
        )
        self.particles = [self.particles[i] for i in indices]
        for i, particle in enumerate(self.particles):
            self.weights[i] *= self._observation_likelihood(particle, observation)
        self.weights /= np.sum(self.weights)
    def _observation_likelihood(self, particle, observation) -> float:
        """Calculate likelihood of observation given particle state."""
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
        if isinstance(action_probs, dict):
            probs = torch.zeros(N_ACTIONS)
            for action, prob in action_probs.items():
                probs[action] = prob
            action_probs = probs
        valid_actions_mask = torch.zeros_like(action_probs)
        for action in self.valid_actions:
            valid_actions_mask[action] = 1
        masked_probs = action_probs * valid_actions_mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs = valid_actions_mask / valid_actions_mask.sum()
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
        self.temperature = temperature
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
    def search(self, state: Dict) -> Dict[int, float]:
        """Perform MCTS from the given root state."""
        root = MCTSNode(state)
        state_tensor = {
            'hand_matrix': state['hand_matrix'].unsqueeze(0).to(self.device),
            'discard_history': state['discard_history'].unsqueeze(0).to(self.device),
            'valid_actions_mask': state['valid_actions_mask'].unsqueeze(0).to(self.device)
        }
        with torch.no_grad():
            if self.policy_network:
                policy_probs = self.policy_network(
                    state_tensor['hand_matrix'],
                    state_tensor['discard_history']
                )[0].squeeze()
            else:
                policy_probs = torch.ones(N_ACTIONS, device=self.device) / N_ACTIONS
        valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
        if isinstance(valid_actions, int):
            valid_actions = [valid_actions]
        root.expand(policy_probs, valid_actions)
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            while node.is_expanded():
                child, action = node.select_child()
                node = child
                search_path.append(node)
            if not node.is_expanded() and len(search_path) < 50:  # Prevent too deep searches
                node_state = node.state
                node_tensor = {
                    'hand_matrix': node_state['hand_matrix'].unsqueeze(0).to(self.device),
                    'discard_history': node_state['discard_history'].unsqueeze(0).to(self.device),
                    'valid_actions_mask': node_state['valid_actions_mask'].unsqueeze(0).to(self.device)
                }
                with torch.no_grad():
                    if self.policy_network:
                        node_policy_probs = self.policy_network(
                            node_tensor['hand_matrix'],
                            node_tensor['discard_history']
                        )[0].squeeze()
                    else:
                        node_policy_probs = torch.ones(N_ACTIONS, device=self.device) / N_ACTIONS
                node_valid_actions = torch.nonzero(node_state['valid_actions_mask']).squeeze().tolist()
                if isinstance(node_valid_actions, int):
                    node_valid_actions = [node_valid_actions]
                node.expand(node_policy_probs, node_valid_actions)
            if self.value_network:
                with torch.no_grad():
                    value = self.value_network(
                        state_tensor['hand_matrix'],
                        state_tensor['discard_history']
                    ).item()
            else:
                value = self._rollout(node.state)
            for node in reversed(search_path):
                node.update(value)
                value = -value  # Flip value for opponent's perspective
        visits = np.array([root.children[a].visit_count if a in root.children else 0 for a in valid_actions])
        if self.temperature == 0:
            best_action = valid_actions[np.argmax(visits)]
            probs = {a: 1.0 if a == best_action else 0.0 for a in valid_actions}
        else:
            visits = np.power(visits, 1.0 / self.temperature)
            visits = visits / np.sum(visits)
            probs = {valid_actions[i]: visits[i] for i in range(len(valid_actions))}
        return probs
    def _rollout(self, state: Dict) -> float:
        """Perform a random rollout from the given state."""
        max_steps = 50  # Prevent infinite rollouts
        current_state = state
        done = False
        steps = 0
        while not done and steps < max_steps:
            valid_actions = torch.nonzero(current_state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
            action = np.random.choice(valid_actions)
            return np.random.uniform(-1, 1)
            steps += 1
        return 0.0  # Default value for timeout
class MCTSValueNetwork(nn.Module):
    """Neural network to predict state values for MCTS."""
    def __init__(self, hidden_size: int = 128):
        super(MCTSValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(52, hidden_size, batch_first=True)
        self.opponent_fc = nn.Linear(52, hidden_size // 2)
        conv_out_size = 32 * 4 * 13
        self.fc1 = nn.Linear(conv_out_size + hidden_size + hidden_size // 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.value_head = nn.Linear(hidden_size // 2, 1)
    def forward(self, hand_matrix, discard_history, opponent_model):
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(hand_matrix.size(0), -1)  # Flatten
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take last LSTM output
        x3 = F.relu(self.opponent_fc(opponent_model))
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = torch.tanh(self.value_head(x))  # Value between -1 and 1
        return value
class MCTSPolicyNetwork(nn.Module):
    """Neural network to predict action probabilities for MCTS."""
    def __init__(self, hidden_size: int = 128):
        super(MCTSPolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(52, hidden_size, batch_first=True)
        self.opponent_fc = nn.Linear(52, hidden_size // 2)
        conv_out_size = 32 * 4 * 13
        self.fc1 = nn.Linear(conv_out_size + hidden_size + hidden_size // 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.policy_head = nn.Linear(hidden_size // 2, N_ACTIONS)
    def forward(self, hand_matrix, discard_history, opponent_model, valid_actions_mask):
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(hand_matrix.size(0), -1)  # Flatten
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take last LSTM output
        x3 = F.relu(self.opponent_fc(opponent_model))
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.policy_head(x)
        action_logits[~valid_actions_mask] = float('-inf')
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
            self.device = torch.device("mps")
    def select_action(self, state: Dict, temperature: float = 1.0, timeout: float = 5.0) -> int:
        """Select an action based on MCTS search with a timeout."""
        self.mcts.temperature = temperature
        start_time = time.time()
        state_device = {
            'hand_matrix': state['hand_matrix'].to(self.device),
            'discard_history': state['discard_history'].to(self.device),
            'valid_actions_mask': state['valid_actions_mask'].to(self.device)
        }
        try:
            import signal
            import platform
            if platform.system() != "Windows":
                def timeout_handler(signum, frame):
                    raise TimeoutError("MCTS search timed out")
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))
            action_probs = self.mcts.search(state_device)
            if platform.system() != "Windows":
                signal.alarm(0)
        except TimeoutError:
            print("MCTS search timed out, using policy network directly")
            if self.policy_network:
                with torch.no_grad():
                    policy_probs = self.policy_network(
                        state_device['hand_matrix'].unsqueeze(0),
                        state_device['discard_history'].unsqueeze(0)
                    )[0].squeeze()
                    policy_probs = policy_probs * state_device['valid_actions_mask']
                    policy_probs = policy_probs / (policy_probs.sum() + 1e-8)
                    valid_actions = torch.nonzero(state_device['valid_actions_mask']).squeeze().tolist()
                    if isinstance(valid_actions, int):
                        valid_actions = [valid_actions]
                    action_probs = {action: float(policy_probs[action]) for action in valid_actions}
            else:
                valid_actions = torch.nonzero(state_device['valid_actions_mask']).squeeze().tolist()
                if isinstance(valid_actions, int):
                    valid_actions = [valid_actions]
                action_probs = {action: 1.0 / len(valid_actions) for action in valid_actions}
        actions, probs = zip(*action_probs.items())
        probs = np.array(probs)
        probs = probs / probs.sum()  # Normalize
        if temperature == 0:
            action = actions[np.argmax(probs)]
        else:
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
        self.mcts.policy_network = self.policy_network
        self.mcts.value_network = self.value_network 
