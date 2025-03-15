import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import namedtuple
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN
from torch.distributions import Categorical

N_ACTIONS = 110
GAMMA = 0.99
ENTROPY_BETA = 0.01

Step = namedtuple('Step', ['state', 'action', 'reward', 'next_state', 'done'])

class LSTM(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        self.wx = nn.Parameter(torch.Tensor(hidden_dim, in_dim))
        self.wh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim // 4))
        self.bx = nn.Parameter(torch.Tensor(hidden_dim))
        self.bh = nn.Parameter(torch.Tensor(hidden_dim))
        
        self.reset()
        
    def reset(self):
        nn.init.uniform_(self.wx, -0.1, 0.1)
        nn.init.uniform_(self.wh, -0.1, 0.1)
        nn.init.zeros_(self.bx)
        nn.init.zeros_(self.bh)
            
    def forward(self, x):
        batch, seq_len, _ = x.size()
        h = torch.zeros(batch, self.hidden_dim // 4, device=x.device)
        c = torch.zeros(batch, self.hidden_dim // 4, device=x.device)
        
        out = []
        for t in range(seq_len):
            gates = x[:, t] @ self.wx.t() + self.bx
            gates = gates + h @ self.wh.t() + self.bh
            
            i, f, g, o = gates.chunk(4, dim=1)
            
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            
            c = f * c + i * g
            h = o * torch.tanh(c)
            
            out.append(h)
            
        return torch.stack(out, dim=1), (h, c)

class Net(nn.Module):
    def __init__(self, n_actions=110):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)
        
        self.apply(self._init)
        
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
            
    def forward(self, cards, history):
        batch = cards.size(0)
        
        if cards.dim() == 3:
            cards = cards.unsqueeze(1)
            
        x = self.conv(cards.float())
        x = x.view(batch, -1)
        
        h, _ = self.lstm(history.float())
        h = h[:, -1]
        
        x = torch.cat([x, h], dim=1)
        x = self.fc(x)
            
        pi = F.softmax(self.policy(x), dim=1)
        v = self.value(x)
        
        return pi, v

class Agent:
    def __init__(self, net=None, lr=3e-4):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")
            
        self.net = net if net else Net()
        self.net.to(self.device)
        
        self.optim = optim.Adam(self.net.parameters(), lr=lr, eps=1e-5)
        self.memory = []
        
        self.gamma = GAMMA
        self.beta = ENTROPY_BETA
        self.max_grad = 0.5
        self.value_weight = 0.25
        
    def act(self, state, explore=True):
        with torch.no_grad():
            state = {k: v.to(self.device) for k, v in state.items()}
            
            pi, _ = self.net(state['hand_matrix'], state['discard_history'])
            pi = pi.squeeze()
            
            mask = state['valid_actions_mask']
            pi = pi * mask
            pi = pi / (pi.sum() + 1e-10)
            
            temp = 0.5 if not explore else 1.0
            pi = F.softmax(torch.log(pi + 1e-10) / temp, dim=-1)
            
            if not explore:
                return torch.argmax(pi).item()
                
            try:
                dist = Categorical(pi)
                return dist.sample().item()
            except:
                return torch.argmax(pi).item()
                
    def store(self, state, action, reward, next_state, done):
        self.memory.append(Step(state, action, reward, next_state, done))
        
    def get_returns(self, rewards):
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns, device=self.device)
        
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
        return returns
        
    def learn(self):
        if not self.memory:
            return {'policy_loss': 0, 'value_loss': 0, 'entropy': 0}
            
        states = [s.state for s in self.memory]
        actions = torch.tensor([s.action for s in self.memory], device=self.device)
        rewards = [s.reward for s in self.memory]
        
        returns = self.get_returns(rewards)
        
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        values = []
        for s in states:
            _, v = self.net(s['hand_matrix'].to(self.device), 
                           s['discard_history'].to(self.device))
            values.append(v)
            
        values = torch.cat(values)
        advantages = returns - values.detach()
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
        batch = len(states)
        for t, (s, a, adv, ret) in enumerate(zip(states, actions, advantages, returns)):
            pi, v = self.net(s['hand_matrix'].to(self.device),
                            s['discard_history'].to(self.device))
                            
            mask = s['valid_actions_mask'].to(self.device)
            pi = pi.squeeze() * mask
            pi = pi / (pi.sum() + 1e-10)
            
            log_pi = torch.log(pi[a] + 1e-10)
            policy_loss += -log_pi * adv.detach()
            
            value_loss += F.mse_loss(v.squeeze(), ret)
            
            entropy = -(pi * torch.log(pi + 1e-10)).sum()
            entropy = torch.clamp(entropy, -2.0, 2.0)
            entropy_loss += entropy
            
        policy_loss /= batch
        value_loss /= batch
        entropy_loss /= batch
        
        loss = (
            policy_loss +
            self.value_weight * value_loss -
            self.beta * entropy_loss
        )
        
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
        self.optim.step()
        
        self.memory.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy_loss.item()
        }
        
    def save(self, path):
        torch.save(self.net.state_dict(), path)
        
    def load(self, path):
        self.net.load_state_dict(torch.load(path, map_location=self.device))
        self.net.eval()

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
        probs = self.net(state_batch)
        eps = 1e-10
        probs = probs.clamp(min=eps, max=1.0 - eps)
        log_probs = torch.log(probs)
        selected_log_probs = log_probs.gather(1, action_batch.unsqueeze(1))
        scaled_rewards = reward_batch * self.reward_scale
        policy_loss = -(selected_log_probs * scaled_rewards.unsqueeze(1)).mean()
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy = torch.clamp(entropy, min=-2.0, max=2.0)  # Prevent extreme entropy values
        loss = policy_loss - self.entropy_coef * entropy
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optim.step()
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
