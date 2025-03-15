import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN
from collections import deque, namedtuple
import random
from typing import List, Tuple, Dict

# Constants
N_SUITS = 4
N_RANKS = 13
N_ACTIONS = 110  # 52 discards + 52 draws + 6 special actions
BATCH_SIZE = 128
BUFFER_SIZE = 50000
TARGET_UPDATE = 1000
GAMMA = 0.99

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQNetwork(nn.Module):
    """Deep Q-Network for Gin Rummy."""
    def __init__(self, state_dim=(1, 4, 13), action_dim=110, hidden_dim=128):
        super(DQNetwork, self).__init__()
        
        # Convolutional layers for processing hand matrix
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # Layer normalization for flattened features
        self.layer_norm = nn.LayerNorm(64 * 4 * 13)
        
        # LSTM for processing discard history
        self.lstm = nn.LSTM(52, 128, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 4 * 13 + 128, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # Action bias
        self.action_bias = nn.Parameter(torch.zeros(1, action_dim))
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, hand_matrix, discard_history, valid_actions_mask=None):
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
        x = self.conv3(x)
        x = F.relu(x)
        
        # Flatten the output
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Apply layer normalization
        x = self.layer_norm(x)
        
        # Process discard history through LSTM
        discard_history = discard_history.float()
        # Reshape discard history if needed
        if discard_history.dim() == 4:  # If [batch, seq_len, height, width]
            discard_history = discard_history.view(batch_size, -1, 52)  # Reshape to [batch, seq_len, features]
        
        lstm_out, _ = self.lstm(discard_history)
        
        # Take the last output from the LSTM
        lstm_out = lstm_out[:, -1, :]
        
        # Concatenate features
        combined = torch.cat([x, lstm_out], dim=1)
        
        # Pass through fully connected layers
        x = self.fc1(combined)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        
        # Apply action bias
        q_values = x + self.action_bias
        
        # Mask invalid actions if mask is provided
        if valid_actions_mask is not None:
            q_values = q_values.clone()
            # Ensure mask has the same shape as q_values
            if valid_actions_mask.dim() == 1 and q_values.dim() > 1:
                valid_actions_mask = valid_actions_mask.unsqueeze(0)
                if valid_actions_mask.size(1) != q_values.size(1):
                    # Handle case where dimensions don't match
                    return q_values
                valid_actions_mask = valid_actions_mask.expand_as(q_values)
            q_values[~valid_actions_mask.bool()] = float('-inf')
        
        return q_values

    def load(self, filepath):
        """Load model checkpoint from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']

class ReplayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)
    
    def push(self, *, state: Dict[str, torch.Tensor], action: int, reward: float, next_state: Dict[str, torch.Tensor], done: bool):
        """Add experience to buffer using named arguments."""
        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        return len(self.buffer)

class DQNAgent:
    """DQN agent for Gin Rummy."""
    
    def __init__(self, state_dim=(1, 4, 13), action_dim=110, hidden_dim=128):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        
        self.model = DQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_model = DQNetwork(state_dim=state_dim, action_dim=action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0
        self.current_loss = None
        
    def select_action(self, state, epsilon=0.1):
        """Select an action using epsilon-greedy policy."""
        if random.random() < epsilon:
            # Random action
            valid_actions = torch.nonzero(state['valid_actions_mask']).squeeze().tolist()
            if isinstance(valid_actions, int):
                valid_actions = [valid_actions]
            return random.choice(valid_actions)
        
        with torch.no_grad():
            # Move state to device and ensure correct shapes
            state_device = {
                'hand_matrix': state['hand_matrix'].unsqueeze(0).to(self.device),
                'discard_history': state['discard_history'].unsqueeze(0).to(self.device),
                'valid_actions_mask': state['valid_actions_mask'].unsqueeze(0).to(self.device)
            }
            
            # Get Q-values
            q_values = self.model(
                state_device['hand_matrix'],
                state_device['discard_history'],
                state_device['valid_actions_mask']
            )
            
            # Mask invalid actions
            masked_q_values = q_values.clone()
            masked_q_values[~state_device['valid_actions_mask'].bool()] = float('-inf')
            
            return masked_q_values.squeeze(0).argmax().item()
    
    def optimize_model(self):
        """Perform one step of optimization."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample experiences
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.state]).to(self.device),
            'valid_actions_mask': torch.stack([s['valid_actions_mask'] for s in batch.state]).to(self.device)
        }
        
        next_state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.next_state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.next_state]).to(self.device),
            'valid_actions_mask': torch.stack([s['valid_actions_mask'] for s in batch.next_state]).to(self.device)
        }
        
        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)
        done_batch = torch.tensor(batch.done, device=self.device, dtype=torch.float32)
        
        # Compute Q(s_t, a)
        state_action_values = self.model(
            state_batch['hand_matrix'],
            state_batch['discard_history'],
            state_batch['valid_actions_mask']
        ).gather(1, action_batch)
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            # Get Q-values from target network
            next_q_values = self.target_model(
                next_state_batch['hand_matrix'],
                next_state_batch['discard_history'],
                next_state_batch['valid_actions_mask']
            )
            
            # Mask invalid actions
            next_q_values = next_q_values.clone()
            next_q_values[~next_state_batch['valid_actions_mask'].bool()] = float('-inf')
            
            # Get maximum Q-value for each next state
            next_state_values = next_q_values.max(1)[0].detach()
            
            # Set V(s) = 0 for terminal states
            next_state_values = next_state_values * (1 - done_batch)
            
            # Compute expected Q values
            expected_state_action_values = reward_batch + (GAMMA * next_state_values)
        
        # Normalize rewards for stable training
        reward_mean = reward_batch.mean()
        reward_std = reward_batch.std() + 1e-6
        normalized_rewards = (reward_batch - reward_mean) / reward_std
        
        # Compute Huber loss
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1)
        )
        
        # Store loss for logging
        self.current_loss = loss.item()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Update target network periodically
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        
        self.steps_done += 1
        
        return self.current_loss
        
    def save(self, filepath):
        """Save model checkpoint to file."""
        checkpoint = {
            'policy_state_dict': self.model.state_dict(),
            'target_state_dict': self.target_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }
        torch.save(checkpoint, filepath)
        
    def load(self, filepath):
        """Load model checkpoint from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['policy_state_dict'])
        self.target_model.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']

    def load_model(self, path):
        """Load model from path."""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.eval() 