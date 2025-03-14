import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import math
import random

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
    """
    Combined Policy and Value Network for MCTS as described in the CS229 milestone.
    
    Architecture:
    - Shared convolutional layers for processing the hand matrix
    - LSTM for processing discard history
    - Separate policy and value heads
    
    This network is inspired by AlphaZero's architecture, adapted for Gin Rummy.
    """
    def __init__(self, hidden_size: int = 256):
        super(PolicyValueNetwork, self).__init__()
        
        # Shared layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Batch normalization for stability
        self.bn2 = nn.BatchNorm2d(64)
        
        # LSTM for discard history
        self.lstm = nn.LSTM(52, hidden_size, batch_first=True)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 4 * 13 + hidden_size, N_ACTIONS)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 4 * 13 + hidden_size, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, hand_matrix, discard_history):
        """
        Forward pass through the network.
        
        Args:
            hand_matrix: Tensor of shape (batch_size, 1, 4, 13) representing the player's hand
            discard_history: Tensor of shape (batch_size, seq_len, 52) representing the discard history
            
        Returns:
            policy_probs: Action probabilities
            value: Estimated state value
        """
        # Process hand through conv layers
        x = F.relu(self.bn1(self.conv1(hand_matrix)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Process discard history
        lstm_out, _ = self.lstm(discard_history)
        lstm_features = lstm_out[:, -1, :]  # Take last LSTM output
        
        # Policy head
        policy_features = F.relu(self.policy_bn(self.policy_conv(x)))
        policy_features = policy_features.view(-1, 32 * 4 * 13)
        policy_features = torch.cat([policy_features, lstm_features], dim=1)
        policy_features = self.dropout(policy_features)
        policy_logits = self.policy_fc(policy_features)
        policy_probs = F.softmax(policy_logits, dim=1)
        
        # Value head
        value_features = F.relu(self.value_bn(self.value_conv(x)))
        value_features = value_features.view(-1, 32 * 4 * 13)
        value_features = torch.cat([value_features, lstm_features], dim=1)
        value_features = self.dropout(value_features)
        value_hidden = F.relu(self.value_fc1(value_features))
        value = torch.tanh(self.value_fc2(value_hidden))
        
        return policy_probs, value

class ParticleFilter:
    """
    Particle filter for opponent modeling in imperfect information games.
    
    This maintains a belief distribution over possible opponent hands
    and updates it based on observed actions.
    """
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
        sum_weights = np.sum(self.weights)
        if sum_weights > 0:
            self.weights /= sum_weights
        else:
            # If all weights are zero, reset to uniform
            self.weights = np.ones(self.n_particles) / self.n_particles
        
    def _observation_likelihood(self, particle, observation) -> float:
        """Calculate likelihood of observation given particle state."""
        # Implementation depends on specific game observations
        pass

class MCTSNode:
    """
    Node in the Monte Carlo Tree Search.
    
    Each node represents a game state and stores:
    - Visit counts
    - Total value
    - Prior probabilities from policy network
    - Children nodes
    """
    def __init__(self, state, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.is_expanded = False
        
    def expand(self, policy_probs: torch.Tensor, valid_actions: List[int]):
        """
        Expand node by adding children for all valid actions.
        
        Args:
            policy_probs: Probabilities from policy network
            valid_actions: List of valid action indices
        """
        for action in valid_actions:
            if action not in self.children:
                # Create child with prior probability from policy network
                prior_prob = policy_probs[0, action].item()
                self.children[action] = MCTSNode(
                    state=None,  # State will be set when child is visited
                    parent=self,
                    action=action,
                    prior_prob=prior_prob
                )
        self.is_expanded = True
        
    def select_child(self) -> Tuple['MCTSNode', int]:
        """
        Select child node using PUCT algorithm.
        
        PUCT balances exploration and exploitation by combining:
        - UCB exploration term
        - Prior probabilities from policy network
        - Value estimates from previous visits
        
        Returns:
            Tuple of (selected child node, action)
        """
        # PUCT formula: Q(s,a) + C_PUCT * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        best_score = float('-inf')
        best_action = -1
        best_child = None
        
        # Parent visit count for exploration bonus
        parent_visit_count = self.visit_count
        
        for action, child in self.children.items():
            # Exploitation term: average value
            if child.visit_count > 0:
                q_value = child.value_sum / child.visit_count
            else:
                q_value = 0.0
                
            # Exploration term
            u_value = C_PUCT * child.prior_prob * math.sqrt(parent_visit_count) / (1 + child.visit_count)
            
            # PUCT score
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
                
        return best_child, best_action
        
    def update(self, value: float):
        """
        Update node statistics after a simulation.
        
        Args:
            value: Value from simulation
        """
        self.visit_count += 1
        self.value_sum += value

class MCTS:
    """
    Monte Carlo Tree Search implementation for Gin Rummy as described in the CS229 milestone.
    
    This implementation uses:
    - Neural network policy and value functions
    - PUCT for node selection
    - Particle filtering for opponent modeling
    - Progressive widening for large action spaces
    """
    def __init__(self, policy_value_net: PolicyValueNetwork, n_iterations: int = N_ITERATIONS, c_puct: float = C_PUCT):
        self.policy_value_net = policy_value_net
        self.n_iterations = n_iterations
        self.c_puct = c_puct
        self.device = next(policy_value_net.parameters()).device
        
    def search(self, root_state: Dict) -> int:
        """
        Perform MCTS search from root state and return best action.
        
        Args:
            root_state: Current game state
            
        Returns:
            Best action according to search
        """
        # Create root node
        root = MCTSNode(state=root_state)
        
        # Get valid actions and policy from neural network
        valid_actions = self._get_valid_actions(root_state)
        hand_matrix = self._create_hand_matrix(root_state['hand']).to(self.device)
        discard_history = self._create_discard_history(root_state['discard_pile']).to(self.device)
        
        with torch.no_grad():
            policy_probs, _ = self.policy_value_net(hand_matrix, discard_history)
        
        # Expand root node
        root.expand(policy_probs, valid_actions)
        
        # Perform simulations
        for _ in range(self.n_iterations):
            # Selection phase: traverse tree to leaf node
            node = root
            search_path = [node]
            
            while node.is_expanded and node.children:
                child, action = node.select_child()
                node = child
                
                # If child state is not set, simulate action to get new state
                if node.state is None:
                    parent_state = search_path[-1].state
                    node.state = self._simulate_action(parent_state, action)
                
                search_path.append(node)
            
            # Expansion and evaluation phase
            if not node.is_expanded and not self._is_terminal(node.state):
                # Get valid actions and policy from neural network
                valid_actions = self._get_valid_actions(node.state)
                hand_matrix = self._create_hand_matrix(node.state['hand']).to(self.device)
                discard_history = self._create_discard_history(node.state['discard_pile']).to(self.device)
                
                with torch.no_grad():
                    policy_probs, value = self.policy_value_net(hand_matrix, discard_history)
                
                # Expand node
                node.expand(policy_probs, valid_actions)
                
                # Use value from neural network
                value = value.item()
            else:
                # For terminal nodes, use game outcome
                if self._is_terminal(node.state):
                    value = self._get_terminal_value(node.state)
                else:
                    # Perform random rollout for non-expanded nodes
                    value = self._simulate(node.state)
            
            # Backpropagation phase
            self._backpropagate(search_path, value)
        
        # Select best action based on visit counts
        visit_counts = [(action, child.visit_count) for action, child in root.children.items()]
        best_action = max(visit_counts, key=lambda x: x[1])[0]
        
        return best_action
    
    def _get_valid_actions(self, state: Dict) -> List[int]:
        """Get list of valid actions for the current state."""
        valid_actions = []
        
        # Draw actions (0-1)
        if state.get('can_draw', True):
            valid_actions.append(0)  # Draw from stock
            if state.get('discard_pile') and len(state.get('discard_pile', [])) > 0:
                valid_actions.append(1)  # Draw from discard
        
        # Discard actions (2-53)
        hand = state.get('hand', [])
        for card in hand:
            valid_actions.append(2 + card)  # Discard card
        
        # Special actions
        if self._can_knock(state):
            valid_actions.append(108)  # Knock
        
        if self._can_gin(state):
            valid_actions.append(109)  # Gin
        
        return valid_actions
    
    def _simulate_action(self, state: Dict, action: int) -> Dict:
        """Simulate taking an action and return the new state."""
        # Create a deep copy of the state
        new_state = {k: v.copy() if isinstance(v, list) else v for k, v in state.items()}
        
        # Process action
        if action == 0:  # Draw from stock
            if new_state.get('stock_pile'):
                drawn_card = new_state['stock_pile'].pop(0)
                new_state['hand'].append(drawn_card)
                new_state['can_draw'] = False
        
        elif action == 1:  # Draw from discard
            if new_state.get('discard_pile') and len(new_state.get('discard_pile', [])) > 0:
                drawn_card = new_state['discard_pile'].pop()
                new_state['hand'].append(drawn_card)
                new_state['can_draw'] = False
        
        elif 2 <= action <= 53:  # Discard
            card = action - 2
            if card in new_state['hand']:
                new_state['hand'].remove(card)
                new_state['discard_pile'].append(card)
                new_state['can_draw'] = True
        
        elif action == 108:  # Knock
            new_state['knocked'] = True
            new_state['terminal'] = True
        
        elif action == 109:  # Gin
            new_state['gin'] = True
            new_state['terminal'] = True
        
        return new_state
    
    def _simulate(self, state: Dict) -> float:
        """
        Perform a random rollout from the current state.
        
        Args:
            state: Current game state
            
        Returns:
            Final value of the simulation
        """
        # Create a copy of the state for simulation
        sim_state = {k: v.copy() if isinstance(v, list) else v for k, v in state.items()}
        
        # Simulate until terminal state
        while not self._is_terminal(sim_state):
            valid_actions = self._get_valid_actions(sim_state)
            if not valid_actions:
                break
                
            # Choose random action
            action = random.choice(valid_actions)
            sim_state = self._simulate_action(sim_state, action)
        
        # Return final value
        return self._get_terminal_value(sim_state)
    
    def _is_terminal(self, state: Dict) -> bool:
        """Check if state is terminal."""
        return state.get('terminal', False) or state.get('knocked', False) or state.get('gin', False)
    
    def _get_terminal_value(self, state: Dict) -> float:
        """Get value for terminal state."""
        if state.get('gin', False):
            return 1.0
        elif state.get('knocked', False):
            # Value depends on deadwood difference
            player_deadwood = self._calculate_deadwood_points(state['hand'])
            opponent_deadwood = state.get('opponent_deadwood', 30)  # Estimate if unknown
            
            if player_deadwood < opponent_deadwood:
                return 0.7  # Win by knocking
            else:
                return -0.5  # Lose by knocking with higher deadwood
        else:
            # Estimate value based on current hand quality
            deadwood = self._calculate_deadwood_points(state['hand'])
            return max(0, (50 - deadwood) / 50)  # Scale between 0 and 1
    
    def _calculate_deadwood_points(self, hand: List[int]) -> int:
        """Calculate deadwood points for a hand."""
        # Find melds
        melds = self._find_melds(hand)
        
        # Calculate deadwood
        melded_cards = set()
        for meld in melds:
            melded_cards.update(meld)
        
        deadwood = 0
        for card in hand:
            if card not in melded_cards:
                rank = card % 13
                # Face cards (J, Q, K) are worth 10 points
                points = min(10, rank + 1)
                deadwood += points
        
        return deadwood
    
    def _find_melds(self, hand: List[int]) -> List[List[int]]:
        """Find all valid melds in a hand."""
        melds = []
        
        # Check for sets (same rank, different suits)
        rank_groups = defaultdict(list)
        for card in hand:
            rank = card % 13
            rank_groups[rank].append(card)
        
        for rank, cards in rank_groups.items():
            if len(cards) >= 3:
                melds.append(cards)
        
        # Check for runs (same suit, consecutive ranks)
        suit_groups = defaultdict(list)
        for card in hand:
            suit = card // 13
            suit_groups[suit].append(card)
        
        for suit, cards in suit_groups.items():
            cards.sort(key=lambda x: x % 13)
            
            # Find runs of 3 or more consecutive cards
            run = []
            for card in cards:
                rank = card % 13
                if not run or rank == (run[-1] % 13) + 1:
                    run.append(card)
                else:
                    if len(run) >= 3:
                        melds.append(run[:])
                    run = [card]
            
            if len(run) >= 3:
                melds.append(run)
        
        return melds
    
    def _create_hand_matrix(self, hand: List[int]) -> torch.Tensor:
        """Convert hand to matrix representation."""
        matrix = torch.zeros(1, 1, 4, 13)
        for card in hand:
            suit = card // 13
            rank = card % 13
            matrix[0, 0, suit, rank] = 1
        return matrix
    
    def _create_discard_history(self, discard_pile: List[int], max_len: int = 52) -> torch.Tensor:
        """Convert discard pile to sequence representation."""
        history = torch.zeros(1, max_len, 52)
        for i, card in enumerate(discard_pile[-max_len:]):
            if i < max_len and 0 <= card < 52:
                history[0, i, card] = 1
        return history
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """Backpropagate value through search path."""
        # Reverse value for opponent's turn
        for node in reversed(search_path):
            node.update(value)
            value = -value  # Negate value for opponent's perspective

class MCTSAgent:
    """
    MCTS Agent for Gin Rummy as described in the CS229 milestone.
    
    This agent uses Monte Carlo Tree Search with neural network policy and value functions
    to select actions in the Gin Rummy environment.
    """
    def __init__(self, policy_value_net=None, num_simulations: int = 100, temperature: float = 1.0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
            
        # Initialize policy-value network
        if policy_value_net is None:
            self.policy_value_net = PolicyValueNetwork().to(self.device)
        else:
            self.policy_value_net = policy_value_net.to(self.device)
            
        # Initialize MCTS
        self.mcts = MCTS(self.policy_value_net, n_iterations=num_simulations)
        self.num_simulations = num_simulations
        self.temperature = temperature
        
    def select_action(self, state: Dict, temperature: float = None) -> int:
        """
        Select action using MCTS.
        
        Args:
            state: Current game state
            temperature: Temperature parameter for exploration (higher = more exploration)
            
        Returns:
            Selected action
        """
        if temperature is None:
            temperature = self.temperature
            
        # Perform MCTS search
        action = self.mcts.search(state)
        return action
    
    def save(self, filepath_prefix: str) -> None:
        """Save policy-value network to file."""
        torch.save(self.policy_value_net.state_dict(), f"{filepath_prefix}_policy_value_net.pt")
        print(f"Saved model to {filepath_prefix}_policy_value_net.pt")
    
    def load(self, filepath_prefix: str) -> None:
        """Load policy-value network from file."""
        self.policy_value_net.load_state_dict(
            torch.load(f"{filepath_prefix}_policy_value_net.pt", map_location=self.device)
        )
        print(f"Loaded model from {filepath_prefix}_policy_value_net.pt") 