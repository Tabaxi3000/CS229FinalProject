import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN
from collections import deque, namedtuple
import random

N_ACTIONS = 110
BUFFER_SIZE = 50000
BATCH_SIZE = 128
GAMMA = 0.99
TARGET_UPDATE = 1000

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    def __init__(self, input_shape=(1, 4, 13), hidden_size=128):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        self.norm = nn.LayerNorm(64 * 4 * 13)
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, N_ACTIONS)
        )
        
        self.action_bias = nn.Parameter(torch.zeros(1, N_ACTIONS))
        
    def forward(self, cards, history, mask=None):
        batch = cards.size(0)
        
        if cards.dim() == 3:
            cards = cards.unsqueeze(1)
            
        x = self.conv(cards.float())
        x = x.view(batch, -1)
        x = self.norm(x)
        
        history = history.float()
        if history.dim() == 4:
            history = history.view(batch, -1, 52)
            
        lstm_out, _ = self.lstm(history)
        lstm_out = lstm_out[:, -1]
        
        x = torch.cat([x, lstm_out], dim=1)
        x = self.head(x) + self.action_bias
        
        if mask is not None:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            x[~mask.bool()] = float('-inf')
            
        return x

class ReplayBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append(Experience(state, action, reward, next_state, done))
        
    def sample(self, size):
        return random.sample(self.buffer, size)
        
    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, input_shape=(1, 4, 13), hidden_size=128):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else
                                 "cuda" if torch.cuda.is_available() else "cpu")
            
        self.q_net = DQN(input_shape, hidden_size).to(self.device)
        self.target_net = DQN(input_shape, hidden_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.optim = optim.Adam(self.q_net.parameters(), lr=1e-4)
        
        self.eps = 1.0
        self.eps_min = 0.01
        self.eps_decay = 0.995
        
        self.steps = 0
        self.last_loss = None
        
    def act(self, state, explore=True):
        if explore and random.random() < self.eps:
            valid = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid, int):
                valid = [valid]
            return random.choice(valid)
            
        with torch.no_grad():
            state = {k: v.unsqueeze(0).to(self.device) for k, v in state.items()}
            q_vals = self.q_net(state['hand_matrix'], 
                              state['discard_history'],
                              state['valid_actions_mask'])
            return q_vals.squeeze(0).argmax().item()
            
    def learn(self):
        if len(self.memory) < BATCH_SIZE:
            return
            
        batch = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*batch))
        
        state = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.state]).to(self.device),
            'valid_actions_mask': torch.stack([s['valid_actions_mask'] for s in batch.state]).to(self.device)
        }
        
        next_state = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.next_state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.next_state]).to(self.device),
            'valid_actions_mask': torch.stack([s['valid_actions_mask'] for s in batch.next_state]).to(self.device)
        }
        
        actions = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        done = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        q_vals = self.q_net(state['hand_matrix'],
                           state['discard_history'],
                           state['valid_actions_mask']).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_net(next_state['hand_matrix'],
                                   next_state['discard_history'],
                                   next_state['valid_actions_mask'])
            next_q[~next_state['valid_actions_mask'].bool()] = float('-inf')
            next_q = next_q.max(1)[0] * (1 - done)
            target = rewards + GAMMA * next_q
            
        loss = F.smooth_l1_loss(q_vals, target.unsqueeze(1))
        self.last_loss = loss.item()
        
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optim.step()
        
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
            
        if self.steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            
        self.steps += 1
        return self.last_loss
        
    def save(self, path):
        torch.save({
            'model': self.q_net.state_dict(),
            'target': self.target_net.state_dict(),
            'optim': self.optim.state_dict(),
            'steps': self.steps,
            'eps': self.eps
        }, path)
        
    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data['model'])
        self.target_net.load_state_dict(data['target'])
        self.optim.load_state_dict(data['optim'])
        self.steps = data['steps']
        self.eps = data['eps'] 