import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import namedtuple
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN
from torch.distributions import Categorical

TOTAL_ACTIONS = 110
DISCOUNT = 0.99
ENTROPY_WEIGHT = 0.01

GameStep = namedtuple('GameStep', ['state', 'action', 'reward', 'next_state', 'done'])

class MemoryCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        self.input_weights = nn.Parameter(torch.Tensor(hidden_dim, input_dim))
        self.hidden_weights = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim // 4))
        self.input_bias = nn.Parameter(torch.Tensor(hidden_dim))
        self.hidden_bias = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.reset_weights()
        
    def reset_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
            
    def forward(self, inputs):
        batch_size, seq_len, _ = inputs.size()
        
        hidden = torch.zeros(batch_size, self.hidden_dim // 4, device=inputs.device)
        cell = torch.zeros(batch_size, self.hidden_dim // 4, device=inputs.device)
        
        outputs = []
        for t in range(seq_len):
            x = inputs[:, t, :]
            
            gates = torch.matmul(x, self.input_weights.t()) + self.input_bias
            gates += torch.matmul(hidden, self.hidden_weights.t()) + self.hidden_bias
            
            i, f, g, o = gates.chunk(4, dim=1)
            
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            
            cell = f * cell + i * g
            hidden = o * torch.tanh(cell)
            
            outputs.append(hidden)
            
        return torch.stack(outputs, dim=1), (hidden, cell)

class GinRummyNet(nn.Module):
    def __init__(self, num_actions=110):
        super().__init__()
        
        self.card_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.history_processor = nn.LSTM(52, 128, batch_first=True)
        
        self.decision_maker = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.move_predictor = nn.Linear(128, num_actions)
        self.value_estimator = nn.Linear(128, 1)
        
        self.apply(self._init_network)
        
    def _init_network(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
            
    def forward(self, cards, history):
        batch_size = cards.size(0)
        
        if cards.dim() == 3:
            cards = cards.unsqueeze(1)
            
        encoded_cards = self.card_encoder(cards.float())
        flattened_cards = encoded_cards.view(batch_size, -1)
        
        history_features, _ = self.history_processor(history.float())
        last_state = history_features[:, -1, :]
        
        features = torch.cat([flattened_cards, last_state], dim=1)
        
        for layer in self.decision_maker:
            features = layer(features)
            
        move_probs = F.softmax(self.move_predictor(features), dim=1)
        state_value = self.value_estimator(features)
        
        return move_probs, state_value

class GinRummyAgent:
    def __init__(self, network=None, learning_rate=0.0003):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.network = network if network else GinRummyNet()
        self.network.to(self.device)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate, eps=1e-5)
        self.memory = []
        
        self.discount = DISCOUNT
        self.entropy_weight = 0.001
        self.max_grad = 0.5
        self.value_weight = 0.25
        
    def choose_action(self, game_state, explore=True):
        with torch.no_grad():
            state_tensor = {
                'hand_matrix': game_state['hand_matrix'].to(self.device),
                'discard_history': game_state['discard_history'].to(self.device),
                'valid_actions_mask': game_state['valid_actions_mask'].to(self.device)
            }
            
            move_probs, _ = self.network(state_tensor['hand_matrix'], state_tensor['discard_history'])
            move_probs = move_probs.squeeze()
            
            legal_moves = state_tensor['valid_actions_mask']
            move_probs = move_probs * legal_moves
            move_probs = move_probs / (move_probs.sum() + 1e-10)
            
            temperature = 0.5 if not explore else 1.0
            move_probs = F.softmax(torch.log(move_probs + 1e-10) / temperature, dim=-1)
            
            if not explore:
                return torch.argmax(move_probs).item()
                
            try:
                distribution = Categorical(move_probs)
                return distribution.sample().item()
            except:
                return torch.argmax(move_probs).item()
                
    def remember(self, state, action, reward, next_state, done):
        experience = GameStep(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.memory.append(experience)
        
    def compute_returns(self, rewards):
        returns = []
        future_return = 0
        
        for r in reversed(rewards):
            future_return = r + self.discount * future_return
            returns.insert(0, future_return)
            
        returns = torch.tensor(returns, device=self.device)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
        return returns
        
    def learn(self):
        if not self.memory:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
            
        states = [step.state for step in self.memory]
        actions = torch.tensor([step.action for step in self.memory], device=self.device)
        rewards = [step.reward for step in self.memory]
        
        returns = self.compute_returns(rewards)
        
        policy_loss = torch.tensor(0.0, device=self.device)
        value_loss = torch.tensor(0.0, device=self.device)
        entropy_loss = torch.tensor(0.0, device=self.device)
        
        state_values = []
        for state in states:
            _, value = self.network(state['hand_matrix'].to(self.device), 
                                  state['discard_history'].to(self.device))
            state_values.append(value)
            
        state_values = torch.cat(state_values)
        advantages = returns - state_values.detach()
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        batch_size = len(states)
        for t, (state, action, advantage, ret) in enumerate(zip(states, actions, advantages, returns)):
            move_probs, value = self.network(state['hand_matrix'].to(self.device),
                                           state['discard_history'].to(self.device))
                                           
            legal_moves = state['valid_actions_mask'].to(self.device)
            move_probs = move_probs.squeeze() * legal_moves
            move_probs = move_probs / (move_probs.sum() + 1e-10)
            
            log_prob = torch.log(move_probs[action] + 1e-10)
            policy_loss += -log_prob * advantage.detach()
            
            value_loss += F.mse_loss(value.squeeze(), ret.unsqueeze(0))
            
            entropy = -(move_probs * torch.log(move_probs + 1e-10)).sum()
            entropy = torch.clamp(entropy, -2.0, 2.0)
            entropy_loss += entropy
            
        policy_loss /= batch_size
        value_loss /= batch_size
        entropy_loss /= batch_size
        
        total_loss = (
            policy_loss +
            self.value_weight * value_loss -
            self.entropy_weight * entropy_loss
        )
        
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad)
        self.optimizer.step()
        
        self.memory.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item()
        }
        
    def save_model(self, path):
        torch.save(self.network.state_dict(), path)
        
    def load_model(self, path):
        self.network.load_state_dict(torch.load(path, map_location=self.device))
        self.network.eval()

    def _find_melds(self, hand_matrix: torch.Tensor) -> List[List[int]]:
        """Find all valid melds in the hand."""
        hand = []
        for suit in range(4):
            for rank in range(13):
                if hand_matrix[0, suit, rank] == 1:
                    hand.append(suit * 13 + rank)
        melds = []
        for rank in range(13):
            same_rank = [card for card in hand if card % 13 == rank]
            if len(same_rank) >= 3:
                if len(same_rank) >= 4:
                    melds.append(same_rank)  # All 4 cards
                for i in range(len(same_rank)):
                    for j in range(i + 1, len(same_rank)):
                        for k in range(j + 1, len(same_rank)):
                            melds.append([same_rank[i], same_rank[j], same_rank[k]])
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
        hand = []
        for suit in range(4):
            for rank in range(13):
                if hand_matrix[0, suit, rank] == 1:
                    hand.append(suit * 13 + rank)
        unmelded = set(hand)
        for meld in melds:
            for card in meld:
                unmelded.discard(card)
        deadwood = 0
        for card in unmelded:
            rank = card % 13
            points = min(10, rank + 1)
            deadwood += points
        return deadwood

    def _create_hand_matrix(self, cards: List[int]) -> np.ndarray:
        """Convert list of card indices to 4x13 matrix."""
        matrix = np.zeros((4, 13), dtype=np.float32)
        for card_idx in cards:
            suit = card_idx // 13
            rank = card_idx % 13
            matrix[suit, rank] = 1
        return matrix[None, ...]  # Add batch dimension

    def _create_discard_history(self, discards: List[int], max_len: int = 52) -> np.ndarray:
        """Convert discard pile to sequence of rank vectors."""
        history = np.zeros((max_len, 13), dtype=np.float32)
        for i, card_idx in enumerate(discards[-max_len:]):
            if i < max_len:  # Ensure we don't exceed max_len
                rank = card_idx % 13
                history[i, rank] = 1.0
        return history[None, ...]  # Add batch dimension

    def _create_opponent_model(self, cards: List[int]) -> np.ndarray:
        """Convert list of known opponent cards to probability vector."""
        model = np.zeros(52, dtype=np.float32)
        for card_idx in cards:
            model[card_idx] = 1
        return model[None, ...]  # Add batch dimension

    def update(self, state_batch, action_batch, reward_batch):
        """Update policy network using policy gradient."""
        probs = self.network(state_batch)
        eps = 1e-10
        probs = probs.clamp(min=eps, max=1.0 - eps)
        log_probs = torch.log(probs)
        selected_log_probs = log_probs.gather(1, action_batch.unsqueeze(1))
        scaled_rewards = reward_batch * self.reward_scale
        policy_loss = -(selected_log_probs * scaled_rewards.unsqueeze(1)).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy = torch.clamp(entropy, min=-2.0, max=2.0)  # Prevent extreme entropy values
        loss = policy_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def calculate_deadwood_and_melds(self, hand):
        """Calculate deadwood points and find melds in a hand."""
        if not hand:
            return float('inf'), []
        sorted_hand = sorted(hand)
        melds = []
        rank_groups = {}
        for card in sorted_hand:
            rank = card % 13
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)
        for rank, cards in rank_groups.items():
            if len(cards) >= 3:
                for i in range(3, min(5, len(cards) + 1)):
                    for combo in self._combinations(cards, i):
                        melds.append(list(combo))
        suit_groups = {}
        for card in sorted_hand:
            suit = card // 13
            if suit not in suit_groups:
                suit_groups[suit] = []
            suit_groups[suit].append(card)
        for suit, cards in suit_groups.items():
            cards.sort(key=lambda x: x % 13)  # Sort by rank within suit
            i = 0
            while i < len(cards):
                run = [cards[i]]
                j = i + 1
                while j < len(cards) and cards[j] % 13 == (cards[j-1] % 13) + 1:
                    run.append(cards[j])
                    j += 1
                if len(run) >= 3:
                    melds.append(run)
                i = j
        if not melds:
            return sum(min(10, x % 13 + 1) for x in hand), []
        min_deadwood = float('inf')
        best_meld_combo = []
        for meld_combo in self._powerset(melds):
            if not meld_combo:
                continue
            used_cards = set()
            valid_combo = True
            for meld in meld_combo:
                if any(card in used_cards for card in meld):
                    valid_combo = False
                    break
                used_cards.update(meld)
            if valid_combo:
                unmatched = [card for card in hand if card not in used_cards]
                deadwood = sum(min(10, x % 13 + 1) for x in unmatched)
                if deadwood < min_deadwood:
                    min_deadwood = deadwood
                    best_meld_combo = meld_combo
        return min_deadwood, best_meld_combo

    def _combinations(self, iterable, r):
        """Helper function to generate combinations."""
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)

    def _powerset(self, iterable):
        """Helper function to generate all possible combinations of melds."""
        s = list(iterable)
        return [combo for r in range(len(s) + 1) for combo in self._combinations(s, r)] 
