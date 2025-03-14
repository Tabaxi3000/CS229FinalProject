import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    def __init__(self, hidden_size: int = 256):
        super(DQNetwork, self).__init__()
        
        # Convolutional layers for hand matrix (4x13)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # LSTM for discard history
        self.lstm = nn.LSTM(52, hidden_size, num_layers=2, batch_first=True)
        
        # Fully connected layers
        conv_out_size = 64 * 4 * 13
        self.fc1 = nn.Linear(conv_out_size + hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, N_ACTIONS)
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, hand_matrix, discard_history, valid_actions_mask):
        # Process hand matrix through conv layers
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = x1.view(hand_matrix.size(0), -1)  # Flatten
        
        # Process discard history through LSTM
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take last LSTM output
        
        # Combine features
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Get Q-values and mask invalid actions
        q_values = self.fc3(x)
        q_values = torch.where(valid_actions_mask, q_values, torch.tensor(float('-inf')).to(q_values.device))
        
        return q_values

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
    def __init__(self, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        
        self.steps_done = 0
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.current_loss = None
        
    def select_action(self, state: Dict[str, torch.Tensor], valid_actions_mask: torch.Tensor) -> int:
        """Select action using epsilon-greedy policy."""
        sample = random.random()
        eps_threshold = max(0.1, 1.0 - self.steps_done * 2e-5)  # Linear decay
        
        if sample > eps_threshold:
            with torch.no_grad():
                q_values = self.policy_net(
                    state['hand_matrix'],
                    state['discard_history'],
                    valid_actions_mask
                )
                return q_values.max(1)[1].item()
        else:
            # Random action from valid actions
            valid_actions = torch.where(valid_actions_mask)[0]
            return random.choice(valid_actions).item()
    
    def optimize_model(self):
        """Perform one step of optimization."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Sample experiences
        experiences = self.memory.sample(BATCH_SIZE)
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors
        state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.state]),
            'discard_history': torch.cat([s['discard_history'] for s in batch.state]),
            'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch.state])
        }
        
        next_state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.next_state]),
            'discard_history': torch.cat([s['discard_history'] for s in batch.next_state]),
            'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch.next_state])
        }
        
        action_batch = torch.tensor(batch.action).to(self.device)
        reward_batch = torch.tensor(batch.reward).to(self.device)
        done_batch = torch.tensor(batch.done).to(self.device)
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(
            state_batch['hand_matrix'],
            state_batch['discard_history'],
            state_batch['valid_actions_mask']
        ).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        with torch.no_grad():
            next_state_values = self.target_net(
                next_state_batch['hand_matrix'],
                next_state_batch['discard_history'],
                next_state_batch['valid_actions_mask']
            ).max(1)[0]
            
            # Set V(s) = 0 for terminal states
            next_state_values[done_batch] = 0
        
        # Compute expected Q values
        expected_state_action_values = reward_batch + GAMMA * next_state_values
        
        # Compute loss
        loss = F.smooth_l1_loss(
            state_action_values,
            expected_state_action_values.unsqueeze(1)
        )
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        
        # Update target network
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.steps_done += 1
        
    def save(self, filepath):
        """Save model checkpoint to file."""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }
        torch.save(checkpoint, filepath)
        
    def load(self, filepath):
        """Load model checkpoint from file."""
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done'] 