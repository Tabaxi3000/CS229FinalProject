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
    def __init__(self, hidden_size: int = 256):
        super(PolicyValueNetwork, self).__init__()
        
        # Shared layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # LSTM for discard history
        self.lstm = nn.LSTM(52, hidden_size, batch_first=True)
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 4 * 13 + hidden_size, N_ACTIONS)
        
        # Value head
        self.value_conv = nn.Conv2d(64, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 4 * 13 + hidden_size, hidden_size)
        self.value_fc2 = nn.Linear(hidden_size, 1)
        
    def forward(self, hand_matrix, discard_history):
        # Process hand through conv layers
        x = F.relu(self.conv1(hand_matrix))
        x = F.relu(self.conv2(x))
        
        # Process discard history
        lstm_out, _ = self.lstm(discard_history)
        lstm_features = lstm_out[:, -1, :]  # Take last LSTM output
        
        # Policy head
        policy_features = F.relu(self.policy_conv(x))
        policy_features = policy_features.view(-1, 32 * 4 * 13)
        policy_features = torch.cat([policy_features, lstm_features], dim=1)
        policy_logits = self.policy_fc(policy_features)
        policy_probs = F.softmax(policy_logits, dim=1)
        
        # Value head
        value_features = F.relu(self.value_conv(x))
        value_features = value_features.view(-1, 32 * 4 * 13)
        value_features = torch.cat([value_features, lstm_features], dim=1)
        value_hidden = F.relu(self.value_fc1(value_features))
        value = torch.tanh(self.value_fc2(value_hidden))
        
        return policy_probs, value

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
        
    def expand(self, policy_probs: torch.Tensor, valid_actions: List[int]):
        """Expand node with children according to policy network."""
        self.valid_actions = valid_actions
        
        # Apply progressive widening - only keep top K actions
        action_probs = policy_probs[valid_actions].numpy()
        top_k_actions = np.argsort(action_probs)[-MAX_CHILDREN:]
        
        for action in top_k_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(
                    state=None,  # Will be set when child is visited
                    parent=self,
                    action=action
                )
                self.children[action].prior_probability = float(policy_probs[action])
    
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
    def __init__(self, policy_value_net: PolicyValueNetwork, n_iterations: int = N_ITERATIONS):
        self.policy_value_net = policy_value_net
        self.n_iterations = n_iterations
        self.particle_filter = ParticleFilter(N_PARTICLES)
        
    def search(self, root_state: Dict) -> int:
        """Perform MCTS search and return best action."""
        root = MCTSNode(state=root_state)
        
        for _ in range(self.n_iterations):
            node = root
            search_path = [node]
            
            # Selection
            while node.children and node.valid_actions:
                node, action = node.select_child()
                search_path.append(node)
            
            # Expansion
            if not node.children and node.valid_actions is None:
                # Get policy and value predictions
                policy_probs, value = self.policy_value_net(
                    node.state['hand_matrix'].unsqueeze(0),
                    node.state['discard_history'].unsqueeze(0)
                )
                valid_actions = self._get_valid_actions(node.state)
                node.expand(policy_probs[0], valid_actions)
                
                # Backpropagate
                self._backpropagate(search_path, float(value[0]))
            
            # Simulation (if needed)
            elif node.children:
                value = self._simulate(node.state)
                self._backpropagate(search_path, value)
        
        # Select action with highest visit count
        return max(root.children.items(),
                  key=lambda item: item[1].visit_count)[0]
    
    def _get_valid_actions(self, state: Dict) -> List[int]:
        """Get list of valid actions for given state."""
        valid_actions = []
        
        # Always valid to draw face down (action 1)
        valid_actions.append(1)
        
        # Check if we can draw face up card (action 0)
        if state.get('can_draw_faceup', True):  # True unless it's 3rd turn with first face up card
            valid_actions.append(0)
        
        # All cards in hand can be discarded (actions 2-53)
        hand_matrix = state['hand_matrix'][0, 0]  # Remove batch and channel dims
        for suit in range(4):
            for rank in range(13):
                if hand_matrix[suit, rank] == 1:
                    valid_actions.append(2 + suit * 13 + rank)
        
        # Check if we can knock or gin (actions 108-109)
        melds = self._find_melds(state['hand_matrix'])
        deadwood = self._calculate_deadwood(state['hand_matrix'], melds)
        
        if deadwood == 0:
            valid_actions.append(109)  # Gin
        elif deadwood <= 10:  # MAX_DEADWOOD in Java
            valid_actions.append(108)  # Knock
        
        return valid_actions
    
    def _simulate(self, state: Dict) -> float:
        """Run simulation to estimate state value."""
        # Use particle filter to sample opponent hand
        opponent_hand = self._sample_opponent_hand()
        
        # Initialize simulation state
        sim_state = state.copy()
        sim_reward = 0.0
        depth = 0
        max_depth = 50  # Prevent infinite loops
        
        while depth < max_depth:
            # Get valid actions
            valid_actions = self._get_valid_actions(sim_state)
            
            # Terminal state checks
            if not valid_actions:
                break
                
            # Check for gin or knock
            if 109 in valid_actions:  # Can go gin
                sim_reward = 1.0  # Maximum reward
                break
            elif 108 in valid_actions:  # Can knock
                # Estimate probability of winning based on deadwood
                melds = self._find_melds(sim_state['hand_matrix'])
                our_deadwood = self._calculate_deadwood(sim_state['hand_matrix'], melds)
                opp_melds = self._find_melds(torch.tensor(opponent_hand).reshape(1, 4, 13))
                opp_deadwood = self._calculate_deadwood(torch.tensor(opponent_hand).reshape(1, 4, 13), opp_melds)
                
                if our_deadwood == 0:  # Gin
                    sim_reward = 1.0
                elif our_deadwood < opp_deadwood:  # Win by points
                    sim_reward = 0.7
                else:  # Undercut
                    sim_reward = -0.3
                break
            
            # Choose random action and update state
            action = np.random.choice(valid_actions)
            
            # Update state based on action
            if action <= 1:  # Draw actions
                drawn_card = np.random.randint(52)  # Simplified card drawing
                sim_state['hand_matrix'][0, 0, drawn_card // 13, drawn_card % 13] = 1
            else:  # Discard actions
                card_idx = action - 2
                sim_state['hand_matrix'][0, 0, card_idx // 13, card_idx % 13] = 0
            
            depth += 1
        
        # If no terminal state reached, estimate value based on melds and deadwood
        if depth == max_depth:
            melds = self._find_melds(sim_state['hand_matrix'])
            deadwood = self._calculate_deadwood(sim_state['hand_matrix'], melds)
            sim_reward = max(0, (25 - deadwood) / 25)  # Scale 0 to 1 based on deadwood
        
        return sim_reward
    
    def _find_melds(self, hand_matrix: torch.Tensor) -> List[List[int]]:
        """Find all valid melds in the hand."""
        # Convert hand matrix to card indices
        hand = []
        for suit in range(4):
            for rank in range(13):
                if hand_matrix[0, suit, rank] == 1:
                    hand.append(suit * 13 + rank)
        
        melds = []
        # Check for sets (same rank, different suits)
        for rank in range(13):
            same_rank = [card for card in hand if card % 13 == rank]
            if len(same_rank) >= 3:
                # Add all possible combinations of 3 or 4 cards
                if len(same_rank) >= 4:
                    melds.append(same_rank)  # All 4 cards
                for i in range(len(same_rank)):
                    for j in range(i + 1, len(same_rank)):
                        for k in range(j + 1, len(same_rank)):
                            melds.append([same_rank[i], same_rank[j], same_rank[k]])
        
        # Check for runs (same suit, consecutive ranks)
        for suit in range(4):
            suit_cards = sorted([card for card in hand if card // 13 == suit])
            run = []
            for card in suit_cards:
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
    
    def _calculate_deadwood(self, hand_matrix: torch.Tensor, melds: List[List[int]]) -> int:
        """Calculate deadwood points from unmelded cards."""
        # Convert hand matrix to card indices
        hand = []
        for suit in range(4):
            for rank in range(13):
                if hand_matrix[0, suit, rank] == 1:
                    hand.append(suit * 13 + rank)
        
        # Remove melded cards
        unmelded = set(hand)
        for meld in melds:
            for card in meld:
                unmelded.discard(card)
        
        # Calculate deadwood points
        deadwood = 0
        for card in unmelded:
            rank = card % 13
            # Face cards (11-13) are worth 10 points
            points = min(10, rank + 1)
            deadwood += points
        
        return deadwood
    
    def _sample_opponent_hand(self) -> np.ndarray:
        """Sample an opponent hand from particle filter."""
        if not self.particle_filter.particles:
            # Initialize with random hand if no particles
            return np.random.randint(0, 2, size=(4, 13))
        
        # Sample a particle based on weights
        particle_idx = np.random.choice(
            len(self.particle_filter.particles),
            p=self.particle_filter.weights
        )
        return self.particle_filter.particles[particle_idx]
    
    def _backpropagate(self, search_path: List[MCTSNode], value: float):
        """Backpropagate value through search path."""
        for node in reversed(search_path):
            node.update(value)
            # Apply TD(λ) backup
            value = LAMBDA * value + (1 - LAMBDA) * (node.value_sum / node.visit_count) 

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
    
    def search(self, root_state: Dict) -> Dict[int, float]:
        """Perform MCTS from the given root state."""
        # Create root node
        root = MCTSNode(root_state)
        
        # Perform simulations
        for _ in range(self.num_simulations):
            leaf = self.select(root)
            
            # Check if the game is over at this leaf
            if leaf.state.get('isTerminal', False):
                value = leaf.state.get('reward', 0.0)
            else:
                # Expand the leaf node
                action_probs = self.get_action_probs(leaf.state)
                leaf.expand(action_probs)
                
                # Evaluate the leaf node
                value = self.evaluate(leaf.state)
            
            # Backpropagate the result
            self.backpropagate(leaf, value)
        
        # Return normalized visit counts as action probabilities
        if self.temperature == 0:
            # Play deterministically
            visit_counts = {action: child.visit_count for action, child in root.children.items()}
            best_action = max(visit_counts.items(), key=lambda x: x[1])[0]
            probs = {action: 1.0 if action == best_action else 0.0 for action in visit_counts}
        else:
            # Apply temperature
            visit_counts = {action: child.visit_count ** (1.0 / self.temperature) for action, child in root.children.items()}
            total_count = sum(visit_counts.values())
            probs = {action: count / total_count for action, count in visit_counts.items()}
        
        return probs
    
    def select(self, node: MCTSNode) -> MCTSNode:
        """Select a leaf node to expand."""
        # Traverse the tree to find a leaf node
        while node.is_expanded() and not node.state.get('isTerminal', False):
            child, action = node.select_child()
            node = child
        
        return node
    
    def evaluate(self, state: Dict) -> float:
        """Evaluate a state using the value network."""
        if self.value_network is None:
            # Use random rollout if no value network
            return self.random_rollout(state)
        
        # Convert state to tensors
        hand_matrix = self.create_hand_matrix(state.get('playerHand', [])).unsqueeze(0).unsqueeze(0)
        discard_history = self.create_discard_history(state.get('discardPile', [])).unsqueeze(0)
        opponent_model = self.create_opponent_model(state.get('knownOpponentCards', [])).unsqueeze(0)
        
        # Move to device
        hand_matrix = hand_matrix.to(self.device)
        discard_history = discard_history.to(self.device)
        opponent_model = opponent_model.to(self.device)
        
        # Get value prediction
        with torch.no_grad():
            value = self.value_network(hand_matrix, discard_history, opponent_model).item()
        
        return value
    
    def backpropagate(self, node: MCTSNode, value: float) -> None:
        """Backpropagate the evaluation through the tree."""
        # Update each node in the selection path
        while node is not None:
            node.update(value)
            node = node.parent
            value = -value  # Negate value for opponent's turn
    
    def get_action_probs(self, state: Dict) -> Dict[int, float]:
        """Get action probabilities from the policy network."""
        valid_actions = self.get_valid_actions(state)
        
        if self.policy_network is None:
            # Use uniform probabilities if no policy network
            return {action: 1.0 / len(valid_actions) for action in valid_actions}
        
        # Convert state to tensors
        hand_matrix = self.create_hand_matrix(state.get('playerHand', [])).unsqueeze(0).unsqueeze(0)
        discard_history = self.create_discard_history(state.get('discardPile', [])).unsqueeze(0)
        opponent_model = self.create_opponent_model(state.get('knownOpponentCards', [])).unsqueeze(0)
        
        # Create valid actions mask
        valid_actions_mask = torch.zeros(1, N_ACTIONS, dtype=torch.bool)
        valid_actions_mask[0, valid_actions] = True
        
        # Move to device
        hand_matrix = hand_matrix.to(self.device)
        discard_history = discard_history.to(self.device)
        opponent_model = opponent_model.to(self.device)
        valid_actions_mask = valid_actions_mask.to(self.device)
        
        # Get action probabilities
        with torch.no_grad():
            probs = self.policy_network(hand_matrix, discard_history, opponent_model, valid_actions_mask)[0]
        
        # Convert to dictionary
        action_probs = {action: probs[action].item() for action in valid_actions}
        
        return action_probs
    
    def random_rollout(self, state: Dict) -> float:
        """Perform a random rollout from the given state."""
        if state.get('isTerminal', False):
            return state.get('reward', 0.0)
        
        # Clone state to avoid modifying original
        current_state = {k: v.copy() if isinstance(v, (list, set, dict)) else v for k, v in state.items()}
        
        # Perform random actions until terminal state
        max_steps = 100  # Limit rollout length
        for _ in range(max_steps):
            if current_state.get('isTerminal', False):
                break
            
            # Get valid actions
            valid_actions = self.get_valid_actions(current_state)
            
            # Choose random action
            action = random.choice(valid_actions)
            
            # Apply action
            current_state = self.simulate_action(current_state, action)
        
        # Return reward from terminal state
        return current_state.get('reward', 0.0)
    
    def get_valid_actions(self, state: Dict) -> List[int]:
        """Get valid actions from a state."""
        # In Gin Rummy, valid actions depend on phase and cards
        valid_actions = []
        
        # Check if we're in drawing phase
        if state.get('phase', 'draw') == 'draw':
            # Can draw from stock or discard pile
            valid_actions = [0, 1]  # 0=draw from discard, 1=draw from stock
        else:
            # Discard phase - can discard any card in hand
            hand = state.get('playerHand', [])
            valid_actions = [2 + card_idx for card_idx in hand]  # 2-53 = discard actions
            
            # Can also knock or gin if eligible
            if self.can_knock(state):
                valid_actions.append(108)  # knock action
            if self.can_gin(state):
                valid_actions.append(109)  # gin action
        
        return valid_actions
    
    def simulate_action(self, state: Dict, action: int) -> Dict:
        """Simulate taking an action from a state."""
        # Clone current state
        new_state = {k: v.copy() if isinstance(v, (list, set, dict)) else v for k, v in state.items()}
        
        if action == 0:  # Draw from discard
            # Add face-up card to hand
            face_up_card = new_state.get('faceUpCard', -1)
            if face_up_card >= 0:
                new_state['playerHand'].append(face_up_card)
                new_state['discardPile'] = new_state['discardPile'][:-1]  # Remove top card
            new_state['phase'] = 'discard'
            
        elif action == 1:  # Draw from stock
            # Simulate drawing a random card
            all_cards = set(range(52))
            used_cards = set(new_state.get('playerHand', []) + 
                            new_state.get('knownOpponentCards', []) + 
                            new_state.get('discardPile', []))
            available_cards = list(all_cards - used_cards)
            if available_cards:
                new_card = random.choice(available_cards)
                new_state['playerHand'].append(new_card)
            new_state['phase'] = 'discard'
            
        elif 2 <= action <= 53:  # Discard a card
            card_idx = action - 2
            if card_idx in new_state.get('playerHand', []):
                new_state['playerHand'].remove(card_idx)
                new_state['discardPile'].append(card_idx)
                new_state['faceUpCard'] = card_idx
            new_state['phase'] = 'draw'
            
        elif action == 108:  # Knock
            # End the game with a knock
            new_state['isTerminal'] = True
            new_state['reward'] = 1.0 if self.get_deadwood_points(new_state) <= 10 else -1.0
            
        elif action == 109:  # Gin
            # End the game with a gin
            new_state['isTerminal'] = True
            new_state['reward'] = 1.0
        
        # Update turn number
        new_state['turnNumber'] = new_state.get('turnNumber', 0) + 1
        
        return new_state
    
    def can_knock(self, state: Dict) -> bool:
        """Check if player can knock."""
        return self.get_deadwood_points(state) <= 10
    
    def can_gin(self, state: Dict) -> bool:
        """Check if player can gin."""
        return self.get_deadwood_points(state) == 0
    
    def get_deadwood_points(self, state: Dict) -> int:
        """Calculate deadwood points."""
        hand = state.get('playerHand', [])
        
        # Find all possible melds in the hand
        melds = self.find_melds(hand)
        
        # Calculate deadwood from cards not in melds
        deadwood_cards = set(hand)
        for meld in melds:
            for card in meld:
                if card in deadwood_cards:
                    deadwood_cards.remove(card)
        
        # Calculate points (face cards = 10, ace = 1, others = value)
        points = 0
        for card in deadwood_cards:
            rank = card % 13
            if rank >= 10:  # Jack, Queen, King
                points += 10
            else:  # Ace-10
                points += rank + 1
        
        return points
    
    def find_melds(self, hand: List[int]) -> List[List[int]]:
        """Find all valid melds in a hand."""
        if not hand:
            return []
        
        # Find sets (same rank, different suits)
        ranks = [[] for _ in range(13)]
        for card in hand:
            rank = card % 13
            ranks[rank].append(card)
        
        # Find runs (same suit, consecutive ranks)
        suits = [[] for _ in range(4)]
        for card in hand:
            suit = card // 13
            suits[suit].append(card)
        
        sets = []
        for rank_cards in ranks:
            if len(rank_cards) >= 3:
                sets.append(rank_cards)
        
        runs = []
        for suit_cards in suits:
            if len(suit_cards) >= 3:
                # Sort by rank
                suit_cards.sort(key=lambda x: x % 13)
                # Find consecutive sequences
                run = [suit_cards[0]]
                for i in range(1, len(suit_cards)):
                    if suit_cards[i] % 13 == (suit_cards[i-1] % 13) + 1:
                        run.append(suit_cards[i])
                    else:
                        if len(run) >= 3:
                            runs.append(run.copy())
                        run = [suit_cards[i]]
                if len(run) >= 3:
                    runs.append(run)
        
        return sets + runs
    
    def create_hand_matrix(self, cards: List[int]) -> torch.Tensor:
        """Convert list of card indices to 4x13 matrix."""
        matrix = np.zeros((4, 13), dtype=np.float32)
        for card_idx in cards:
            suit = card_idx // 13
            rank = card_idx % 13
            matrix[suit, rank] = 1
        return torch.FloatTensor(matrix)
    
    def create_discard_history(self, discards: List[int], max_len: int = 52) -> torch.Tensor:
        """Convert discard pile to one-hot sequence."""
        history = np.zeros((max_len, 52), dtype=np.float32)
        for i, card_idx in enumerate(discards[-max_len:]):
            if card_idx >= 0 and i < max_len:  # Skip invalid indices
                history[i, card_idx] = 1
        return torch.FloatTensor(history)
    
    def create_opponent_model(self, cards: List[int]) -> torch.Tensor:
        """Convert list of opponent's known cards to vector."""
        model = np.zeros(52, dtype=np.float32)
        for card_idx in cards:
            model[card_idx] = 1
        return torch.FloatTensor(model)

class MCTSAgent:
    """MCTS agent for Gin Rummy."""
    def __init__(self, policy_network=None, value_network=None, num_simulations: int = 100):
        self.mcts = MCTS(policy_network, value_network, num_simulations)
        self.policy_network = policy_network
        self.value_network = value_network
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
    
    def select_action(self, state: Dict, temperature: float = 1.0) -> int:
        """Select an action based on MCTS search."""
        # Set exploration temperature
        self.mcts.temperature = temperature
        
        # Perform MCTS search
        action_probs = self.mcts.search(state)
        
        # Sample action from probabilities
        actions, probs = zip(*action_probs.items())
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