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
N_SUITS = 4
N_RANKS = 13
N_ACTIONS = 110  # 52 discards + 52 draws + 6 special actions
GAMMA = 0.99  # Discount factor
ENTROPY_COEF = 0.01  # Entropy regularization coefficient
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih_l0 = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh_l0 = nn.Parameter(torch.Tensor(hidden_size, hidden_size // 4))
        self.bias_ih_l0 = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh_l0 = nn.Parameter(torch.Tensor(hidden_size))
        self.weight_ih_l1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size // 4))
        self.weight_hh_l1 = nn.Parameter(torch.Tensor(hidden_size, hidden_size // 4))
        self.bias_ih_l1 = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_hh_l1 = nn.Parameter(torch.Tensor(hidden_size))
        self.reset_parameters()
    def reset_parameters(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.1, 0.1)
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        h_t = torch.zeros(batch_size, self.hidden_size // 4, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size // 4, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            i_t = torch.matmul(x_t, self.weight_ih_l0[:self.hidden_size // 4].t()) + self.bias_ih_l0[:self.hidden_size // 4]
            f_t = torch.matmul(x_t, self.weight_ih_l0[self.hidden_size // 4:self.hidden_size // 2].t()) + self.bias_ih_l0[self.hidden_size // 4:self.hidden_size // 2]
            g_t = torch.matmul(x_t, self.weight_ih_l0[self.hidden_size // 2:3 * self.hidden_size // 4].t()) + self.bias_ih_l0[self.hidden_size // 2:3 * self.hidden_size // 4]
            o_t = torch.matmul(x_t, self.weight_ih_l0[3 * self.hidden_size // 4:].t()) + self.bias_ih_l0[3 * self.hidden_size // 4:]
            i_t += torch.matmul(h_t, self.weight_hh_l0[:self.hidden_size // 4].t()) + self.bias_hh_l0[:self.hidden_size // 4]
            f_t += torch.matmul(h_t, self.weight_hh_l0[self.hidden_size // 4:self.hidden_size // 2].t()) + self.bias_hh_l0[self.hidden_size // 4:self.hidden_size // 2]
            g_t += torch.matmul(h_t, self.weight_hh_l0[self.hidden_size // 2:3 * self.hidden_size // 4].t()) + self.bias_hh_l0[self.hidden_size // 2:3 * self.hidden_size // 4]
            o_t += torch.matmul(h_t, self.weight_hh_l0[3 * self.hidden_size // 4:].t()) + self.bias_hh_l0[3 * self.hidden_size // 4:]
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t)
        layer0_output = torch.stack(outputs, dim=1)
        h_t = torch.zeros(batch_size, self.hidden_size // 4, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_size // 4, device=x.device)
        outputs = []
        for t in range(seq_len):
            x_t = layer0_output[:, t, :]
            i_t = torch.matmul(x_t, self.weight_ih_l1[:self.hidden_size // 4].t()) + self.bias_ih_l1[:self.hidden_size // 4]
            f_t = torch.matmul(x_t, self.weight_ih_l1[self.hidden_size // 4:self.hidden_size // 2].t()) + self.bias_ih_l1[self.hidden_size // 4:self.hidden_size // 2]
            g_t = torch.matmul(x_t, self.weight_ih_l1[self.hidden_size // 2:3 * self.hidden_size // 4].t()) + self.bias_ih_l1[self.hidden_size // 2:3 * self.hidden_size // 4]
            o_t = torch.matmul(x_t, self.weight_ih_l1[3 * self.hidden_size // 4:].t()) + self.bias_ih_l1[3 * self.hidden_size // 4:]
            i_t += torch.matmul(h_t, self.weight_hh_l1[:self.hidden_size // 4].t()) + self.bias_hh_l1[:self.hidden_size // 4]
            f_t += torch.matmul(h_t, self.weight_hh_l1[self.hidden_size // 4:self.hidden_size // 2].t()) + self.bias_hh_l1[self.hidden_size // 4:self.hidden_size // 2]
            g_t += torch.matmul(h_t, self.weight_hh_l1[self.hidden_size // 2:3 * self.hidden_size // 4].t()) + self.bias_hh_l1[self.hidden_size // 2:3 * self.hidden_size // 4]
            o_t += torch.matmul(h_t, self.weight_hh_l1[3 * self.hidden_size // 4:].t()) + self.bias_hh_l1[3 * self.hidden_size // 4:]
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            outputs.append(h_t)
        return torch.stack(outputs, dim=1), (h_t, c_t)
class PolicyNetwork(nn.Module):
    """Policy network for REINFORCE algorithm."""
    def __init__(self, action_space=110):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        self.fc1 = nn.Linear(64 * 4 * 13 + 128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.action_head = nn.Linear(128, action_space)
        self.value_head = nn.Linear(128, 1)
        self.apply(self._init_weights)
    def _init_weights(self, m):
        """Initialize weights for the network."""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
    def forward(self, hand_matrix, discard_history):
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
        combined = torch.cat([x, lstm_out], dim=1)
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        action_probs = F.softmax(self.action_head(x), dim=1)
        state_value = self.value_head(x)
        return action_probs, state_value
    def load(self, filepath):
        """Load model checkpoint from file."""
        checkpoint = torch.load(filepath, map_location=next(self.parameters()).device)
        self.load_state_dict(checkpoint['policy_state_dict'])
class REINFORCEAgent:
    """REINFORCE agent for Gin Rummy with baseline for variance reduction."""
    def __init__(self, model=None, learning_rate=0.0003):  # Reduced learning rate
        self.model = model if model else PolicyNetwork()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps=1e-5)
        self.saved_experiences = []
        self.gamma = 0.99
        self.entropy_coef = 0.001  # Reduced entropy coefficient
        self.max_grad_norm = 0.5
        self.reward_scale = 0.1
        self.value_loss_coef = 0.25  # Added value loss coefficient
    def select_action(self, state, eval_mode=False):
        """Select an action based on policy network."""
        with torch.no_grad():
            hand_matrix = state['hand_matrix'].to(self.device)
            discard_history = state['discard_history'].to(self.device)
            valid_actions_mask = state['valid_actions_mask'].to(self.device)
            action_probs, _ = self.model(hand_matrix, discard_history)
            action_probs = action_probs.squeeze()
            action_probs = action_probs * valid_actions_mask
            action_probs = action_probs / (action_probs.sum() + 1e-10)
            temperature = 0.5 if eval_mode else 1.0
            action_probs = F.softmax(torch.log(action_probs + 1e-10) / temperature, dim=-1)
            if eval_mode:
                action = torch.argmax(action_probs).item()
            else:
                try:
                    m = Categorical(action_probs)
                    action = m.sample().item()
                except Exception as e:
                    print(f"Error sampling action: {e}")
                    action = torch.argmax(action_probs).item()
        return action
    def load_model(self, path):
        """Load model from path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
    def store_experience(self, *, state: Dict[str, torch.Tensor], action: int, reward: float, next_state: Optional[Dict[str, torch.Tensor]], done: bool):
        """Store experience for later training. Uses keyword arguments for clarity."""
        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.saved_experiences.append(experience)
    def calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """Calculate discounted returns with improved reward scaling."""
        returns = []
        R = 0
        scaled_rewards = rewards
        for r in reversed(scaled_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
        return returns
    def calculate_intermediate_reward(self, state: Dict[str, torch.Tensor]) -> float:
        """Calculate intermediate rewards based on game state."""
        return 0.0
    def train(self):
        """Train policy using collected experiences with baseline subtraction."""
        if not self.saved_experiences:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
        states = [exp.state for exp in self.saved_experiences]
        actions = torch.tensor([exp.action for exp in self.saved_experiences], device=self.device)
        rewards = [exp.reward for exp in self.saved_experiences]
        returns = self.calculate_returns(rewards)
        returns = returns.to(self.device)
        total_policy_loss = torch.tensor(0.0, device=self.device)
        total_value_loss = torch.tensor(0.0, device=self.device)
        total_entropy_loss = torch.tensor(0.0, device=self.device)
        state_values = []
        for state in states:
            hand_matrix = state['hand_matrix'].to(self.device)
            discard_history = state['discard_history'].to(self.device)
            _, value = self.model(hand_matrix, discard_history)
            state_values.append(value)
        state_values = torch.cat(state_values)
        advantages = returns - state_values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        batch_size = len(states)
        for t, (state, action, advantage, ret) in enumerate(zip(states, actions, advantages, returns)):
            hand_matrix = state['hand_matrix'].to(self.device)
            discard_history = state['discard_history'].to(self.device)
            probs, value = self.model(hand_matrix, discard_history)
            valid_actions_mask = state['valid_actions_mask'].to(self.device)
            probs = probs.squeeze() * valid_actions_mask
            probs = probs / (probs.sum() + 1e-10)
            log_prob = torch.log(probs[action] + 1e-10)
            policy_loss = -log_prob * advantage.detach()  # Detach advantage
            total_policy_loss = total_policy_loss + policy_loss
            value_target = ret.view(-1)  # Flatten target
            value_pred = value.view(-1)  # Flatten prediction
            value_loss = F.mse_loss(value_pred, value_target)
            total_value_loss = total_value_loss + value_loss
            entropy = -(probs * torch.log(probs + 1e-10)).sum()
            entropy = torch.clamp(entropy, -2.0, 2.0)  # Clip entropy to reasonable range
            total_entropy_loss = total_entropy_loss + entropy
        total_policy_loss = total_policy_loss / batch_size
        total_value_loss = total_value_loss / batch_size
        total_entropy_loss = total_entropy_loss / batch_size
        total_loss = (
            total_policy_loss +
            self.value_loss_coef * total_value_loss -  # Value loss coefficient
            self.entropy_coef * total_entropy_loss  # Entropy coefficient
        )
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.saved_experiences = []
        return {
            'policy_loss': total_policy_loss.item(),
            'value_loss': total_value_loss.item(),
            'entropy': total_entropy_loss.item()
        }
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
        probs = self.model(state_batch)
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()
    def save(self, filepath):
        """Save model checkpoint to file."""
        checkpoint = {
            'policy_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)
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
