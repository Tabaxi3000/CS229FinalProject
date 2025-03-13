import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import namedtuple, deque
from tqdm import tqdm
import os
import random
import json
import time
import numpy as np
from typing import Dict, List, Tuple

# Constants
N_SUITS = 4
N_RANKS = 13
N_ACTIONS = 110  # 52 discards + 52 draws + 6 special actions
BATCH_SIZE = 1024  # Large batch size for efficient training
BUFFER_SIZE = 20000  # Larger buffer
GAMMA = 0.99
ENTROPY_COEF = 0.01

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# ----------------- ENHANCED MODEL ARCHITECTURES -----------------

class EnhancedDQNetwork(nn.Module):
    """Enhanced DQN architecture with increased capacity"""
    def __init__(self, hidden_size: int = 256):  # Increased hidden size
        super(EnhancedDQNetwork, self).__init__()
        
        # Improved CNN layers
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)  # More filters
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Improved LSTM layer with correct input size
        self.lstm = nn.LSTM(13, hidden_size, num_layers=2, batch_first=True, dropout=0.2)
        
        # Expanded fully connected layers
        conv_out_size = 256 * 4 * 13  # Increased size due to more filters
        self.fc1 = nn.Linear(conv_out_size + hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, N_ACTIONS)
        
        # Action biases with stronger initial values
        self.action_bias = nn.Parameter(torch.zeros(N_ACTIONS))
        with torch.no_grad():
            self.action_bias[108] = 1.0  # Stronger bias for knock
            self.action_bias[109] = 2.0  # Even stronger bias for gin
        
        self.dropout = nn.Dropout(0.2)  # Increased dropout
        self.layer_norm = nn.LayerNorm(hidden_size)  # Added layer normalization
        
    def forward(self, hand_matrix, discard_history, valid_actions_mask):
        # Process hand matrix through improved CNN
        x1 = F.leaky_relu(self.conv1(hand_matrix))  # Using LeakyReLU
        x1 = F.leaky_relu(self.conv2(x1))
        x1 = F.leaky_relu(self.conv3(x1))
        x1 = x1.view(hand_matrix.size(0), -1)
        
        # Process discard history through LSTM
        # Reshape discard history to (batch_size, sequence_length, input_size)
        batch_size = discard_history.size(0)
        discard_history = discard_history.view(batch_size, -1, 13)  # Reshape to (batch, seq_len, 13)
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take last LSTM output
        
        # Combine features with layer normalization
        x = torch.cat([x1, x2], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        q_values = self.fc3(x)
        
        # Add action bias and scale for better Q-value separation
        q_values = q_values + self.action_bias
        
        # Apply valid actions mask
        q_values = torch.where(valid_actions_mask, q_values, torch.tensor(float('-inf')).to(q_values.device))
        
        return q_values

class EnhancedPolicyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # CNN for hand matrix
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # LSTM for discard history
        self.lstm = nn.LSTM(input_size=13, hidden_size=64, num_layers=2, batch_first=True)
        
        # Linear layers for opponent model
        self.opponent_fc = nn.Linear(52, 128)
        
        # Combine all features
        self.combine_features = nn.Sequential(
            nn.Linear(320, 256),  # 128 (CNN) + 64 (LSTM) + 128 (opponent) = 320
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 110)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Action biases for game-ending actions
        self.register_parameter('action_biases', 
                              nn.Parameter(torch.zeros(110)))
        self.action_biases.data[108:110] = 0.1  # Small positive bias for knock/gin
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, hand_matrix, discard_history, opponent_model, valid_actions_mask, temperature=1.0):
        # Process hand matrix with CNN
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = F.relu(self.conv3(x1))
        x1 = F.adaptive_avg_pool2d(x1, (1, 1))
        x1 = x1.view(x1.size(0), -1)
        
        # Process discard history with LSTM
        # Reshape discard history to (batch_size, sequence_length, input_size)
        batch_size = discard_history.size(0)
        discard_history = discard_history.view(batch_size, 4, 13)  # Reshape to (batch, 4, 13)
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take the last output
        
        # Process opponent model
        x3 = F.relu(self.opponent_fc(opponent_model))
        
        # Combine features
        combined = torch.cat([x1, x2, x3], dim=1)
        logits = self.combine_features(combined)
        
        # Add action biases
        logits = logits + self.action_biases
        
        # Apply temperature scaling
        logits = logits / temperature
        
        # Mask invalid actions
        if valid_actions_mask is not None:
            logits = logits.masked_fill(~valid_actions_mask, float('-inf'))
        
        # Convert to probabilities
        probs = F.softmax(logits, dim=1)
        
        return probs

# ----------------- ENHANCED AGENTS -----------------

class ReplayBuffer:
    """Replay Buffer with increased capacity"""
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)
        self.terminal_indices = []  # Track indices of terminal states
    
    def push(self, *, state: Dict[str, torch.Tensor], action: int, reward: float, next_state: Dict[str, torch.Tensor], done: bool):
        """Add experience to buffer using named arguments."""
        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.buffer.append(experience)
        
        # Track terminal states (done=True) separately for prioritized sampling
        if done:
            self.terminal_indices.append(len(self.buffer) - 1)
            # Keep terminal_indices in sync with buffer size
            while self.terminal_indices and self.terminal_indices[0] >= len(self.buffer):
                self.terminal_indices.pop(0)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences with emphasis on terminal states."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)
            
        # Prioritize terminal states: select 30% of samples from terminal states if possible
        terminal_ratio = 0.3
        terminal_count = min(int(batch_size * terminal_ratio), len(self.terminal_indices))
        regular_count = batch_size - terminal_count
        
        # Sample from terminal states
        terminal_samples = []
        if terminal_count > 0 and self.terminal_indices:
            terminal_indices = random.sample(self.terminal_indices, terminal_count)
            terminal_samples = [self.buffer[idx] for idx in terminal_indices]
        
        # Sample from all states for the remaining
        regular_indices = random.sample(range(len(self.buffer)), regular_count)
        regular_samples = [self.buffer[idx] for idx in regular_indices]
        
        return terminal_samples + regular_samples
    
    def __len__(self) -> int:
        return len(self.buffer)

class EnhancedDQNAgent:
    """Enhanced DQN agent with improved training features"""
    def __init__(self, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
        
        # Initialize networks
        self.policy_net = EnhancedDQNetwork().to(self.device)
        self.target_net = EnhancedDQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        
        self.steps_done = 0
        self.current_loss = None
        
        # For mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
    
    def optimize_model(self):
        """Perform one step of optimization with mixed precision"""
        if len(self.memory) < 128:  # A bit higher threshold
            return
        
        # Sample experiences
        experiences = self.memory.sample(min(BATCH_SIZE, len(self.memory)))
        if not experiences:
            return
            
        batch = Experience(*zip(*experiences))
        
        # Convert to tensors and move to device
        state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.state]).to(self.device),
            'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch.state]).to(self.device)
        }
        
        next_state_batch = {
            'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.next_state]).to(self.device),
            'discard_history': torch.cat([s['discard_history'] for s in batch.next_state]).to(self.device),
            'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch.next_state]).to(self.device)
        }
        
        action_batch = torch.tensor([a for a in batch.action], device=self.device)
        reward_batch = torch.tensor([r for r in batch.reward], device=self.device)
        done_batch = torch.tensor([d for d in batch.done], device=self.device)
        
        # Use mixed precision if available
        if self.scaler:
            with torch.cuda.amp.autocast():
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
                expected_state_action_values = (reward_batch + GAMMA * next_state_values).unsqueeze(1)
                
                # Compute loss
                loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            
            # Optimize with mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard computation without mixed precision
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
            expected_state_action_values = (reward_batch + GAMMA * next_state_values).unsqueeze(1)
            
            # Compute loss
            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Added gradient clipping
            self.optimizer.step()
        
        self.current_loss = loss.item()
        
        # Update target network - more frequent updates
        if self.steps_done % 200 == 0:
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

class EnhancedREINFORCEAgent:
    """Enhanced REINFORCE agent with improved training features"""
    def __init__(self, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        
        self.policy = EnhancedPolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Increased entropy coefficient for better exploration
        self.entropy_coef = 0.05
        self.value_coef = 0.5
        self.max_grad_norm = 0.5
        
        self.memory = ReplayBuffer(BUFFER_SIZE)
        self.steps_done = 0
        self.current_loss = None
    
    def update(self, state_batch, action_batch, reward_batch):
        """Perform a REINFORCE update with improved stability"""
        # Move tensors to device
        for key in state_batch:
            if not state_batch[key].device == self.device:
                state_batch[key] = state_batch[key].to(self.device)
        
        if not action_batch.device == self.device:
            action_batch = action_batch.to(self.device)
        
        if not reward_batch.device == self.device:
            reward_batch = reward_batch.to(self.device)
        
        # Normalize rewards with improved stability
        if reward_batch.numel() > 1:  # Check if we have more than one reward
            reward_mean = reward_batch.mean()
            reward_std = reward_batch.std()
            if reward_std > 0:
                reward_batch = (reward_batch - reward_mean) / (reward_std + 1e-8)
        
        # Get action probabilities with temperature scaling
        probs = self.policy(
            state_batch['hand_matrix'],
            state_batch['discard_history'],
            state_batch['opponent_model'],
            state_batch['valid_actions_mask'],
            temperature=1.0  # Start with normal temperature
        )
        
        # Calculate log probabilities with improved numerical stability
        log_probs = torch.log(probs + 1e-10)
        
        # Get selected action log probabilities
        action_indices = action_batch.argmax(dim=1)
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Calculate importance weights for different action types
        action_weights = torch.ones_like(reward_batch, device=self.device)
        gin_actions = (action_indices == 109)
        knock_actions = (action_indices == 108)
        draw_actions = (action_indices <= 1)  # Draw from stock or discard
        
        # Higher weights for game-ending actions
        action_weights[gin_actions] = 5.0    # Significantly higher weight for gin
        action_weights[knock_actions] = 3.0  # Higher weight for knock
        action_weights[draw_actions] = 0.8   # Slightly lower weight for draw actions
        
        # Calculate policy gradient loss with improved stability
        weighted_advantages = reward_batch * action_weights
        policy_loss = -(selected_log_probs * weighted_advantages).mean()
        
        # Calculate entropy with clamping to prevent extreme values
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy = torch.clamp(entropy, min=-2.0, max=2.0)
        
        # Total loss with increased entropy coefficient
        loss = policy_loss - self.entropy_coef * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        
        self.optimizer.step()
        
        self.current_loss = loss.item()
        self.steps_done += 1
        
        return loss.item()

# ----------------- DATA HANDLING -----------------

class GinRummyDataset:
    """Dataset class that loads directly from a file"""
    def __init__(self, data_file: str):
        """Initialize dataset from JSON file."""
        self.games = {}
        self.load_data(data_file)
        
        # Calculate dataset statistics
        self.num_games = len(self.games)
        self.num_states = sum(len(states) for states in self.games.values())
    
    def load_data(self, data_file: str):
        """Load game data from JSON file."""
        print(f"Loading data from {data_file}")
        with open(data_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, dict):
                states = data.get('gameStates', [])
            else:
                states = data
            
            # Group states by game ID
            for state in states:
                game_id = state.get('gameId', 0)
                if game_id not in self.games:
                    self.games[game_id] = []
                self.games[game_id].append(state)
    
    def _create_hand_matrix(self, cards: List[int]) -> np.ndarray:
        """Convert list of card indices to 4x13 matrix."""
        matrix = np.zeros((4, 13), dtype=np.float32)
        for card_idx in cards:
            suit = card_idx // 13
            rank = card_idx % 13
            matrix[suit, rank] = 1
        return matrix
    
    def _create_discard_history(self, discards: List[int]) -> np.ndarray:
        """Convert discard pile to history matrix."""
        history = np.zeros((4, 13), dtype=np.float32)
        for card_idx in discards:
            suit = card_idx // 13
            rank = card_idx % 13
            history[suit, rank] = 1
        return history
    
    def get_training_data(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch of training data."""
        # Sample random indices
        indices = np.random.choice(len(self.games), batch_size)
        game_ids = list(self.games.keys())
        selected_games = np.random.choice(game_ids, batch_size)
        
        # Initialize batch arrays
        hand_matrices = []
        discard_histories = []
        opponent_cards = []
        actions = []
        rewards = []
        dones = []
        
        for game_id in selected_games:
            # Randomly select a state from the game
            game_states = self.games[game_id]
            state_idx = np.random.randint(0, len(game_states))
            state = game_states[state_idx]
            
            # Get state data
            hand_matrices.append(self._create_hand_matrix(state['playerHand']))
            discard_histories.append(self._create_discard_history(state['discardPile']))
            opponent_cards.append(self._create_hand_matrix(state['knownOpponentCards']))
            
            # Parse action string to get numeric index
            action_vec = np.zeros(110, dtype=np.float32)
            if state['action'].startswith('draw_faceup_'):
                action_idx = 0 if state['action'].endswith('True') else 1
            elif state['action'].startswith('discard_'):
                try:
                    card_id = int(state['action'].split('_')[1])
                    action_idx = 2 + card_id
                except:
                    action_idx = -1
            elif state['action'] == 'knock':
                action_idx = 108
            elif state['action'] == 'gin':
                action_idx = 109
            else:
                action_idx = -1
            
            if action_idx >= 0 and action_idx < 110:
                action_vec[action_idx] = 1
            actions.append(action_vec)
            
            # Enhanced reward structure
            base_reward = state['reward']
            
            # Calculate deadwood count if available
            deadwood = state.get('deadwoodCount', 30)  # Default to high deadwood if not available
            
            # Adjust reward based on action type and outcome
            if action_idx == 108:  # knock action
                if base_reward > 0:  # Successful knock
                    reward = base_reward * 3.0  # Triple reward for successful knock
                    if deadwood < 5:  # Extra bonus for very low deadwood
                        reward *= 1.5
                else:  # Failed knock
                    reward = base_reward * 1.5  # Increase penalty for failed knock
            
            elif action_idx == 109:  # gin action
                if base_reward > 0:  # Successful gin
                    reward = base_reward * 5.0  # 5x reward for gin
                else:
                    reward = base_reward
            
            else:  # Regular actions
                reward = base_reward
                
                # Small bonus for actions that reduce deadwood
                if deadwood < 10:
                    reward += 0.1
                
                # Small penalty for being in later stages of the game
                # Use state index as a proxy for turn count
                relative_progress = state_idx / len(game_states)
                if relative_progress > 0.5:  # If we're in the later half of the game
                    reward -= 0.05 * (relative_progress - 0.5)
            
            rewards.append(reward)
            dones.append(state['isTerminal'])
        
        # Convert to tensors
        batch = {
            'hand_matrix': torch.FloatTensor(np.array(hand_matrices))[:, None, :, :],
            'discard_history': torch.FloatTensor(np.array(discard_histories)),
            'opponent_model': torch.FloatTensor(np.array(opponent_cards)).reshape(batch_size, -1),
            'valid_actions_mask': torch.ones(batch_size, 110, dtype=torch.bool)
        }
        
        return (batch,
                torch.FloatTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)),
                torch.BoolTensor(np.array(dones)))
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        total_states = sum(len(states) for states in self.games.values())
        total_games = len(self.games)
        return {
            'total_games': total_games,
            'total_states': total_states,
            'avg_game_length': total_states / max(1, total_games)
        }

    def sample_batch(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of training data."""
        # Sample random game IDs and state indices
        game_ids = list(self.games.keys())
        selected_games = np.random.choice(game_ids, batch_size)
        
        # Initialize batch arrays
        hand_matrices = []
        discard_histories = []
        opponent_cards = []
        actions = []
        rewards = []
        dones = []
        
        for game_id in selected_games:
            # Randomly select a state from the game
            game_states = self.games[game_id]
            state_idx = np.random.randint(0, len(game_states))
            state = game_states[state_idx]
            
            # Create hand matrix
            hand_matrix = self._create_hand_matrix(state['playerHand'])
            hand_matrices.append(hand_matrix)
            
            # Create discard history
            discard_history = self._create_discard_history(state['discardPile'])
            discard_histories.append(discard_history)
            
            # Create opponent known cards
            opponent_matrix = self._create_hand_matrix(state['knownOpponentCards'])
            opponent_cards.append(opponent_matrix)
            
            # Parse action string to get numeric index
            action_vec = np.zeros(110, dtype=np.float32)
            if state['action'].startswith('draw_faceup_'):
                action_idx = 0 if state['action'].endswith('True') else 1
            elif state['action'].startswith('discard_'):
                try:
                    card_id = int(state['action'].split('_')[1])
                    action_idx = 2 + card_id
                except:
                    action_idx = -1
            elif state['action'] == 'knock':
                action_idx = 108
            elif state['action'] == 'gin':
                action_idx = 109
            else:
                action_idx = -1
            
            if action_idx >= 0 and action_idx < 110:
                action_vec[action_idx] = 1
            actions.append(action_vec)
            
            # Enhanced reward structure
            base_reward = state['reward']
            
            # Calculate deadwood count if available
            deadwood = state.get('deadwoodCount', 30)  # Default to high deadwood if not available
            
            # Adjust reward based on action type and outcome
            if action_idx == 108:  # knock action
                if base_reward > 0:  # Successful knock
                    reward = base_reward * 3.0  # Triple reward for successful knock
                    if deadwood < 5:  # Extra bonus for very low deadwood
                        reward *= 1.5
                else:  # Failed knock
                    reward = base_reward * 1.5  # Increase penalty for failed knock
            
            elif action_idx == 109:  # gin action
                if base_reward > 0:  # Successful gin
                    reward = base_reward * 5.0  # 5x reward for gin
                else:
                    reward = base_reward
            
            else:  # Regular actions
                reward = base_reward
                
                # Small bonus for actions that reduce deadwood
                if deadwood < 10:
                    reward += 0.1
                
                # Small penalty for being in later stages of the game
                # Use state index as a proxy for turn count
                relative_progress = state_idx / len(game_states)
                if relative_progress > 0.5:  # If we're in the later half of the game
                    reward -= 0.05 * (relative_progress - 0.5)
            
            rewards.append(reward)
            dones.append(state['isTerminal'])
        
        # Convert to tensors
        batch = {
            'hand_matrix': torch.FloatTensor(np.array(hand_matrices))[:, None, :, :],
            'discard_history': torch.FloatTensor(np.array(discard_histories)),
            'opponent_model': torch.FloatTensor(np.array(opponent_cards)).reshape(batch_size, -1),
            'valid_actions_mask': torch.ones(batch_size, 110, dtype=torch.bool)
        }
        
        return (batch,
                torch.FloatTensor(np.array(actions)),
                torch.FloatTensor(np.array(rewards)),
                torch.BoolTensor(np.array(dones)))

def create_enhanced_dataset(input_files, output_file, sample_ratio=0.20):
    """Create a larger dataset with 20% of the games from multiple files."""
    print(f"Creating enhanced dataset with {sample_ratio*100}% of data...")
    
    all_states = []
    all_games = {}
    
    # Load data from multiple files if provided
    if isinstance(input_files, str):
        input_files = [input_files]
    
    for input_file in input_files:
        print(f"Processing {input_file}...")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Group by game ID
        for state in data:
            game_id = state.get('gameId', 0)
            if game_id not in all_games:
                all_games[game_id] = []
            all_games[game_id].append(state)
    
    # Sample a larger subset of games
    game_ids = list(all_games.keys())
    sample_size = max(1, int(len(game_ids) * sample_ratio))
    sampled_ids = random.sample(game_ids, sample_size)
    
    sampled_data = []
    for game_id in sampled_ids:
        sampled_data.extend(all_games[game_id])
    
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f)
    
    print(f"Created enhanced dataset with {len(sampled_ids)} games ({len(sampled_data)} states)")
    return output_file

# ----------------- TRAINING FUNCTIONS -----------------

def train_enhanced_dqn(dataset_file, epochs=10, batch_size=1024, iterations_per_epoch=300):
    """Enhanced DQN training with more epochs and iterations."""
    print("Starting enhanced DQN training...")
    start_time = time.time()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = GinRummyDataset(dataset_file)
    stats = dataset.get_stats()
    print(f"Dataset stats: {stats}")
    
    # Initialize agent
    agent = EnhancedDQNAgent()
    
    # Track best loss for early stopping
    best_loss = float('inf')
    early_stop_patience = 3
    no_improvement_count = 0
    
    # Training loop with more iterations per epoch
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        total_batches = 0
        
        # More iterations per epoch
        progress_bar = tqdm(range(iterations_per_epoch), desc="Training")
        for _ in progress_bar:
            # Get batch of training data
            state_batch, action_batch, reward_batch, done_batch = dataset.get_training_data(batch_size)
            
            # Add experiences to replay buffer
            for i in range(min(128, batch_size)):  # Only use part of batch for faster buffer filling
                agent.memory.push(
                    state={
                        'hand_matrix': state_batch['hand_matrix'][i:i+1],
                        'discard_history': state_batch['discard_history'][i:i+1],
                        'valid_actions_mask': state_batch['valid_actions_mask'][i:i+1]
                    },
                    action=torch.argmax(action_batch[i]).item(),
                    reward=reward_batch[i].item(),
                    next_state={
                        'hand_matrix': state_batch['hand_matrix'][i:i+1],
                        'discard_history': state_batch['discard_history'][i:i+1],
                        'valid_actions_mask': state_batch['valid_actions_mask'][i:i+1]
                    },
                    done=done_batch[i].item()
                )
            
            # Optimize model
            agent.optimize_model()
            
            # Track loss
            current_loss = agent.current_loss or 0
            epoch_loss += current_loss
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': current_loss,
                'memory': len(agent.memory),
                'time': f"{(time.time() - start_time) / 60:.1f}m"
            })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / max(1, total_batches)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.6f}")
        
        # Save checkpoint after each epoch
        model_path = f"models/dqn_enhanced_epoch_{epoch + 1}.pt"
        agent.save(model_path)
        print(f"Saved model checkpoint to {model_path}")
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improvement_count = 0
            # Save best model
            best_model_path = "models/dqn_enhanced_best.pt"
            agent.save(best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} epochs")
            
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Save final model
    agent.save("models/dqn_enhanced_final.pt")
    
    # Print training time
    training_time = (time.time() - start_time) / 60
    print(f"Enhanced DQN training complete! Final model saved. Total training time: {training_time:.2f} minutes")
    return agent

def train_enhanced_reinforce(dataset_file, epochs=5, batch_size=1024, iterations_per_epoch=150):
    """Train the enhanced REINFORCE agent."""
    print("\nStarting enhanced REINFORCE training...")
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading data from {dataset_file}")
    dataset = GinRummyDataset(dataset_file)
    print(f"Loaded {dataset.num_games} games with {dataset.num_states} total states")
    print(f"Dataset stats: {dataset.get_stats()}")
    
    # Initialize agent
    agent = EnhancedREINFORCEAgent()
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        progress_bar = tqdm(range(iterations_per_epoch), desc="Training")
        
        for i in progress_bar:
            # Sample batch from dataset
            state_batch, action_batch, reward_batch, done_batch = dataset.get_training_data(batch_size)
            
            # Update agent
            loss = agent.update(state_batch, action_batch, reward_batch)
            
            # Store experience in replay buffer
            for j in range(batch_size):
                if done_batch[j] or random.random() < 0.3:
                    agent.memory.push(
                        state={
                            'hand_matrix': state_batch['hand_matrix'][j:j+1],
                            'discard_history': state_batch['discard_history'][j:j+1],
                            'opponent_model': state_batch['opponent_model'][j:j+1],
                            'valid_actions_mask': state_batch['valid_actions_mask'][j:j+1]
                        },
                        action=action_batch[j],
                        reward=reward_batch[j].item(),
                        next_state={
                            'hand_matrix': state_batch['hand_matrix'][j:j+1],
                            'discard_history': state_batch['discard_history'][j:j+1],
                            'opponent_model': state_batch['opponent_model'][j:j+1],
                            'valid_actions_mask': state_batch['valid_actions_mask'][j:j+1]
                        },
                        done=done_batch[j].item()
                    )
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss:.3f}",
                'memory': len(agent.memory),
                'time': f"{i/iterations_per_epoch:.1f}m"
            })
        
        # Save checkpoint
        save_checkpoint(agent, epoch, loss, is_best=False)
    
    # Save final model
    save_checkpoint(agent, epochs, loss, is_best=True)

def save_checkpoint(agent, epoch, loss, is_best=False):
    """Save a checkpoint of the model."""
    os.makedirs('models', exist_ok=True)
    
    if is_best:
        checkpoint_path = 'models/reinforce_enhanced_best.pt'
    else:
        checkpoint_path = f'models/reinforce_enhanced_epoch_{epoch + 1}.pt'
    
    checkpoint = {
        'policy_state_dict': agent.policy.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

def main():
    # Create enhanced dataset from multiple files
    input_files = []
    for i in range(1, 4):  # Use first 3 consolidated files
        file_path = f"../java/MavenProject/training_data_consolidated_{i}.json"
        if os.path.exists(file_path):
            input_files.append(file_path)
    
    if not input_files:
        print("No training data files found! Please check path.")
        return
        
    dataset_file = "../java/MavenProject/enhanced_training_data.json"
    
    # Create a dataset with 10% of the games from the input files
    # We're using a smaller dataset for faster training during debugging
    create_enhanced_dataset(input_files, dataset_file, sample_ratio=0.10)
    
    # Train both models in sequence (REINFORCE first, then DQN)
    print("\n\n=== STARTING ENHANCED REINFORCE TRAINING ===\n")
    # Using fewer epochs and iterations for faster training during debugging
    train_enhanced_reinforce(dataset_file, epochs=5, batch_size=1024, iterations_per_epoch=150)
    
    print("\n\n=== STARTING ENHANCED DQN TRAINING ===\n")
    # Using fewer epochs and iterations for faster training during debugging
    train_enhanced_dqn(dataset_file, epochs=5, batch_size=1024, iterations_per_epoch=150)
    
    print("\n\n=== ENHANCED TRAINING COMPLETE ===\n")
    print("Enhanced models are available in the models/ directory")
    print("  - REINFORCE: models/reinforce_enhanced_final.pt")
    print("  - DQN: models/dqn_enhanced_final.pt")
    print("  - Best REINFORCE: models/reinforce_enhanced_best.pt")
    print("  - Best DQN: models/dqn_enhanced_best.pt")

if __name__ == "__main__":
    main() 