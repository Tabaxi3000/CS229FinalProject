import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict

TOTAL_ACTIONS = 110
MEMORY_SIZE = 50000
TRAINING_BATCH = 128
DISCOUNT = 0.99
UPDATE_TARGET_EVERY = 1000

GameExperience = namedtuple('GameExperience', ['state', 'action', 'reward', 'next_state', 'done'])

class GinRummyNet(nn.Module):
    def __init__(self, input_shape=(1, 4, 13), num_actions=110, hidden_size=128):
        super().__init__()
        
        self.card_encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        self.norm = nn.LayerNorm(64 * 4 * 13)
        self.history_processor = nn.LSTM(52, 128, batch_first=True)
        
        self.decision_maker = nn.Sequential(
            nn.Linear(64 * 4 * 13 + 128, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )
        
        self.action_weights = nn.Parameter(torch.zeros(1, num_actions))
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, cards, history, action_mask=None):
        batch = cards.size(0)
        
        if cards.dim() == 3:
            cards = cards.unsqueeze(1)
            
        encoded_cards = self.card_encoder(cards.float())
        flattened = encoded_cards.view(batch, -1)
        normalized = self.norm(flattened)
        
        history = history.float()
        if history.dim() == 4:
            history = history.view(batch, -1, 52)
            
        lstm_output, _ = self.history_processor(history)
        last_hidden = lstm_output[:, -1, :]
        
        features = torch.cat([normalized, last_hidden], dim=1)
        
        for layer in self.decision_maker:
            features = layer(features)
            if isinstance(layer, nn.Linear):
                features = self.dropout(features)
                
        q_values = features + self.action_weights
        
        if action_mask is not None:
            q_values = q_values.clone()
            if action_mask.dim() == 1 and q_values.dim() > 1:
                action_mask = action_mask.unsqueeze(0)
                if action_mask.size(1) != q_values.size(1):
                    return q_values
                action_mask = action_mask.expand_as(q_values)
            q_values[~action_mask.bool()] = float('-inf')
            
        return q_values

class ExperienceBuffer:
    def __init__(self, max_size):
        self.memory = deque(maxlen=max_size)
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(GameExperience(state, action, reward, next_state, done))
        
    def get_batch(self, size):
        return random.sample(self.memory, size)
        
    def __len__(self):
        return len(self.memory)

class GinRummyAgent:
    def __init__(self, input_shape=(1, 4, 13), num_actions=110, hidden_size=128):
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        self.brain = GinRummyNet(input_shape, num_actions, hidden_size).to(self.device)
        self.target_brain = GinRummyNet(input_shape, num_actions, hidden_size).to(self.device)
        self.target_brain.load_state_dict(self.brain.state_dict())
        
        self.memory = ExperienceBuffer(MEMORY_SIZE)
        self.optimizer = optim.Adam(self.brain.parameters(), lr=0.0001)
        
        self.exploration_rate = 1.0
        self.min_exploration = 0.01
        self.exploration_decay = 0.995
        
        self.training_steps = 0
        self.last_loss = None
        
    def choose_action(self, game_state, explore=True):
        if explore and random.random() < self.exploration_rate:
            valid_moves = torch.nonzero(game_state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_moves, int):
                valid_moves = [valid_moves]
            return random.choice(valid_moves)
            
        with torch.no_grad():
            state_tensor = {
                'hand_matrix': game_state['hand_matrix'].unsqueeze(0).to(self.device),
                'discard_history': game_state['discard_history'].unsqueeze(0).to(self.device),
                'valid_actions_mask': game_state['valid_actions_mask'].unsqueeze(0).to(self.device)
            }
            
            q_values = self.brain(
                state_tensor['hand_matrix'],
                state_tensor['discard_history'],
                state_tensor['valid_actions_mask']
            )
            
            masked_values = q_values.clone()
            masked_values[~state_tensor['valid_actions_mask'].bool()] = float('-inf')
            
            return masked_values.squeeze(0).argmax().item()
            
    def learn(self):
        if len(self.memory) < TRAINING_BATCH:
            return
            
        experiences = self.memory.get_batch(TRAINING_BATCH)
        batch = GameExperience(*zip(*experiences))
        
        state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.state]).to(self.device),
            'valid_actions_mask': torch.stack([s['valid_actions_mask'] for s in batch.state]).to(self.device)
        }
        
        next_states = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.next_state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.next_state]).to(self.device),
            'valid_actions_mask': torch.stack([s['valid_actions_mask'] for s in batch.next_state]).to(self.device)
        }
        
        actions = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        done_flags = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        current_q = self.brain(
            state_batch['hand_matrix'],
            state_batch['discard_history'],
            state_batch['valid_actions_mask']
        ).gather(1, actions)
        
        with torch.no_grad():
            next_q = self.target_brain(
                next_states['hand_matrix'],
                next_states['discard_history'],
                next_states['valid_actions_mask']
            )
            
            next_q = next_q.clone()
            next_q[~next_states['valid_actions_mask'].bool()] = float('-inf')
            max_next_q = next_q.max(1)[0].detach()
            max_next_q = max_next_q * (1 - done_flags)
            
            target_q = rewards + (DISCOUNT * max_next_q)
            
        loss = F.smooth_l1_loss(current_q, target_q.unsqueeze(1))
        self.last_loss = loss.item()
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 1.0)
        self.optimizer.step()
        
        if self.exploration_rate > self.min_exploration:
            self.exploration_rate *= self.exploration_decay
            
        if self.training_steps % UPDATE_TARGET_EVERY == 0:
            self.target_brain.load_state_dict(self.brain.state_dict())
            
        self.training_steps += 1
        return self.last_loss
        
    def save_checkpoint(self, filepath):
        checkpoint = {
            'model_state': self.brain.state_dict(),
            'target_model_state': self.target_brain.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'training_steps': self.training_steps,
            'exploration_rate': self.exploration_rate
        }
        torch.save(checkpoint, filepath)
        
    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.brain.load_state_dict(checkpoint['model_state'])
        self.target_brain.load_state_dict(checkpoint['target_model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.training_steps = checkpoint['training_steps']
        self.exploration_rate = checkpoint['exploration_rate'] 