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

TOTAL_ACTIONS = 110
EXPLORATION_CONSTANT = 1.41
DISCOUNT_FACTOR = 0.8
MAX_BRANCHING = 5
PUCT_CONSTANT = 1.0

class GameNetwork(nn.Module):
    def __init__(self, num_actions=110):
        super().__init__()
        
        self.hand_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.history_processor = nn.LSTM(52, 128, batch_first=True)
        self.opponent_encoder = nn.Linear(52, 64)
        
        self.decision_maker = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 128 + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.move_predictor = nn.Linear(128, num_actions)
        self.position_evaluator = nn.Linear(128, 1)
        
    def forward(self, hand, history, opponent_info=None):
        batch_size = hand.size(0)
        
        if hand.dim() == 3:
            hand = hand.unsqueeze(1)
            
        encoded_hand = self.hand_encoder(hand.float())
        flattened_hand = encoded_hand.view(batch_size, -1)
        
        history = history.float()
        history_features, _ = self.history_processor(history)
        last_state = history_features[:, -1, :]
        
        if opponent_info is not None:
            opponent_features = F.relu(self.opponent_encoder(opponent_info.float()))
        else:
            opponent_features = torch.zeros(batch_size, 64, device=hand.device)
            
        features = torch.cat([flattened_hand, last_state, opponent_features], dim=1)
        
        for layer in self.decision_maker:
            features = layer(features)
            
        move_probs = F.softmax(self.move_predictor(features), dim=1)
        position_value = torch.tanh(self.position_evaluator(features))
        
        return move_probs, position_value
        
    def load_weights(self, model_path, device=None):
        if device is None:
            device = next(self.parameters()).device
        self.load_state_dict(torch.load(model_path, map_location=device))

class BeliefTracker:
    def __init__(self, num_particles=100):
        self.num_particles = num_particles
        self.particles = []
        self.weights = np.ones(num_particles) / num_particles
        
    def update_beliefs(self, observation):
        sampled_indices = np.random.choice(
            self.num_particles,
            size=self.num_particles,
            p=self.weights
        )
        
        self.particles = [self.particles[i] for i in sampled_indices]
        
        for i, particle in enumerate(self.particles):
            self.weights[i] *= self._get_likelihood(particle, observation)
            
        self.weights /= np.sum(self.weights)
        
    def _get_likelihood(self, particle, observation):
        return 1.0

class TreeNode:
    def __init__(self, game_state, parent=None, move=None):
        self.game_state = game_state
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.total_value = 0
        self.prior = 0
        self.legal_moves = None
        self.is_leaf = True
        
    def expand(self, move_probabilities, legal_moves=None):
        self.is_leaf = False
        self.legal_moves = legal_moves if legal_moves else list(range(TOTAL_ACTIONS))
        
        if isinstance(move_probabilities, dict):
            probs = torch.zeros(TOTAL_ACTIONS)
            for move, prob in move_probabilities.items():
                probs[move] = prob
            move_probabilities = probs
            
        legal_moves_mask = torch.zeros_like(move_probabilities)
        for move in self.legal_moves:
            legal_moves_mask[move] = 1
            
        masked_probs = move_probabilities * legal_moves_mask
        if masked_probs.sum() > 0:
            masked_probs = masked_probs / masked_probs.sum()
        else:
            masked_probs = legal_moves_mask / legal_moves_mask.sum()
            
        for move in self.legal_moves:
            if masked_probs[move] > 0:
                self.children[move] = TreeNode(
                    game_state=None,
                    parent=self,
                    move=move
                )
                self.children[move].prior = float(masked_probs[move])
                
    def select_best_child(self):
        best_score = float('-inf')
        best_child = None
        best_move = None
        
        for move, child in self.children.items():
            exploration = EXPLORATION_CONSTANT * child.prior * math.sqrt(self.visits)
            exploitation = child.total_value / (child.visits + 1)
            score = exploitation + exploration / (child.visits + 1)
            
            if score > best_score:
                best_score = score
                best_child = child
                best_move = move
                
        return best_child, best_move
        
    def update_stats(self, value):
        self.visits += 1
        self.total_value += value

class MCTS:
    def __init__(self, network=None, num_simulations=100, temperature=1.0):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.network = network
        self.num_simulations = num_simulations
        self.temperature = temperature
        
    def search(self, game_state):
        root = TreeNode(game_state)
        
        state_tensor = {
            'hand_matrix': game_state['hand_matrix'].unsqueeze(0).to(self.device),
            'discard_history': game_state['discard_history'].unsqueeze(0).to(self.device),
            'valid_actions_mask': game_state['valid_actions_mask'].unsqueeze(0).to(self.device)
        }
        
        with torch.no_grad():
            if self.network:
                move_probs, _ = self.network(
                    state_tensor['hand_matrix'],
                    state_tensor['discard_history']
                )
                move_probs = move_probs.squeeze()
            else:
                move_probs = torch.ones(TOTAL_ACTIONS, device=self.device) / TOTAL_ACTIONS
                
        legal_moves = torch.nonzero(game_state['valid_actions_mask']).squeeze().tolist()
        if isinstance(legal_moves, int):
            legal_moves = [legal_moves]
            
        root.expand(move_probs, legal_moves)
        
        for _ in range(self.num_simulations):
            node = root
            path = [node]
            
            while not node.is_leaf:
                child, move = node.select_best_child()
                node = child
                path.append(node)
                
            if node.is_leaf and len(path) < 50:
                node_state = node.game_state
                node_tensor = {
                    'hand_matrix': node_state['hand_matrix'].unsqueeze(0).to(self.device),
                    'discard_history': node_state['discard_history'].unsqueeze(0).to(self.device),
                    'valid_actions_mask': node_state['valid_actions_mask'].unsqueeze(0).to(self.device)
                }
                
                with torch.no_grad():
                    if self.network:
                        node_probs, node_value = self.network(
                            node_tensor['hand_matrix'],
                            node_tensor['discard_history']
                        )
                        node_probs = node_probs.squeeze()
                        value = node_value.item()
                    else:
                        node_probs = torch.ones(TOTAL_ACTIONS, device=self.device) / TOTAL_ACTIONS
                        value = self._simulate(node.game_state)
                        
                node_legal_moves = torch.nonzero(node_state['valid_actions_mask']).squeeze().tolist()
                if isinstance(node_legal_moves, int):
                    node_legal_moves = [node_legal_moves]
                    
                node.expand(node_probs, node_legal_moves)
                
            for node in reversed(path):
                node.update_stats(value)
                value = -value
                
        visit_counts = np.array([root.children[m].visits if m in root.children else 0 for m in legal_moves])
        
        if self.temperature == 0:
            best_move = legal_moves[np.argmax(visit_counts)]
            move_probs = {m: 1.0 if m == best_move else 0.0 for m in legal_moves}
        else:
            visit_counts = np.power(visit_counts, 1.0 / self.temperature)
            visit_counts = visit_counts / np.sum(visit_counts)
            move_probs = {legal_moves[i]: visit_counts[i] for i in range(len(legal_moves))}
            
        return move_probs
        
    def _simulate(self, state):
        steps = 0
        max_steps = 50
        
        while steps < max_steps:
            legal_moves = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(legal_moves, int):
                legal_moves = [legal_moves]
            return np.random.uniform(-1, 1)
            steps += 1
            
        return 0.0

class MCTSAgent:
    def __init__(self, network=None, num_simulations=100):
        self.network = network
        self.num_simulations = num_simulations
        self.mcts = MCTS(network, num_simulations)
        
    def choose_move(self, game_state, temperature=1.0, time_limit=5.0):
        start_time = time.time()
        move_probs = self.mcts.search(game_state)
        
        if time.time() - start_time > time_limit:
            legal_moves = torch.nonzero(game_state['valid_actions_mask']).squeeze().tolist()
            if isinstance(legal_moves, int):
                legal_moves = [legal_moves]
            return random.choice(legal_moves)
            
        moves = list(move_probs.keys())
        probs = list(move_probs.values())
        
        if temperature == 0:
            chosen_move = moves[np.argmax(probs)]
        else:
            chosen_move = np.random.choice(moves, p=probs)
            
        return chosen_move
        
    def save_model(self, path):
        if self.network:
            torch.save(self.network.state_dict(), path)
            
    def load_model(self, path):
        if self.network:
            self.network.load_weights(path) 
