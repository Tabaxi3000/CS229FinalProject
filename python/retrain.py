import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os
import json
import time
from tqdm import tqdm
from typing import Dict, List, Tuple
from collections import namedtuple, deque
import argparse

# Constants
BUFFER_SIZE = 20000
BATCH_SIZE = 1024
GAMMA = 0.99
N_ACTIONS = 110  # 52 discards + 52 draws + 6 special actions
ENTROPY_COEF = 0.01
N_SUITS = 4
N_RANKS = 13

# Define experience tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

print("=== Gin Rummy Model Retraining ===")
print("Initializing retraining process with focus on game-winning actions...")

class GinRummyDQNetwork(nn.Module):
    """DQN network with architecture designed to prioritize winning actions."""
    def __init__(self, hidden_size: int = 192):
        super(GinRummyDQNetwork, self).__init__()
        
        # Convolutional layers for hand processing
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # LSTM for discard history
        self.lstm = nn.LSTM(52, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        
        # Dense layers
        conv_out_size = 32 * 4 * 13
        self.fc1 = nn.Linear(conv_out_size + hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, N_ACTIONS)
        
        # Action bias with stronger initialization for knock/gin
        self.register_parameter('action_bias', nn.Parameter(torch.zeros(N_ACTIONS)))
        with torch.no_grad():
            self.action_bias[108] = 2.0  # Knock - stronger bias
            self.action_bias[109] = 3.0  # Gin - even stronger bias
        
        self.dropout = nn.Dropout(0.2)  # Increased dropout
        
    def forward(self, hand_matrix, discard_history, valid_actions_mask):
        # Process hand matrix
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(hand_matrix.size(0), -1)
        
        # Process discard history
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]
        
        # Combine features
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        
        # Add action bias with scaling factor
        q_values = q_values + self.action_bias * 5.0  # Increased scaling
        
        # Apply valid actions mask
        q_values = torch.where(valid_actions_mask, q_values, torch.tensor(float('-inf')).to(q_values.device))
        
        return q_values

class GinRummyDQNAgent:
    """DQN agent with focus on winning actions."""
    def __init__(self, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
            
        # Initialize networks
        self.policy_net = GinRummyDQNetwork().to(self.device)
        self.target_net = GinRummyDQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE)
        
        self.steps_done = 0
        self.current_loss = 0
        
    def optimize_model(self):
        """Train the model on a batch from the replay buffer."""
        if len(self.memory) < 128:
            return
            
        # Sample batch from replay buffer
        experiences = self.memory.sample(min(BATCH_SIZE, len(self.memory)))
        if not experiences:
            return
            
        batch = Experience(*zip(*experiences))
        
        # Prepare batch data
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
        
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(
            state_batch['hand_matrix'],
            state_batch['discard_history'],
            state_batch['valid_actions_mask']
        ).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1})
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
        
        # Compute loss with extra weight on knock and gin actions
        # Create weights to emphasize knock and gin
        action_weights = torch.ones_like(action_batch, dtype=torch.float)
        knock_mask = (action_batch == 108)
        gin_mask = (action_batch == 109)
        action_weights[knock_mask] = 2.0  # Double importance for knock
        action_weights[gin_mask] = 3.0    # Triple importance for gin
        
        # Apply weighted Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values, reduction='none')
        loss = (loss * action_weights.unsqueeze(1)).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to stabilize training
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        self.current_loss = loss.item()
        
        # Update target network periodically
        if self.steps_done % 200 == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        self.steps_done += 1
        
    def save(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }
        torch.save(checkpoint, filepath)

class GinRummyDataset:
    """Dataset class that loads and processes training data with emphasis on winning actions"""
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.games = {}  # gameId -> list of states
        self.terminal_states = []  # List of terminal states (game endings)
        self.knock_states = []  # List of states with knock actions
        self.gin_states = []  # List of states with gin actions
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the JSON data with special focus on terminal states."""
        print(f"Loading data from {self.json_file}")
        with open(self.json_file, 'r') as f:
            data = json.load(f)
            
            # Extract game states array
            if isinstance(data, dict):
                states = data.get('gameStates', [])
            else:  # data is already a list
                states = data
            
            # Group states by game and identify special states
            for state in states:
                game_id = state.get('gameId', 0)
                if game_id not in self.games:
                    self.games[game_id] = []
                self.games[game_id].append(state)
                
                # Track terminal states
                if state.get('isTerminal', False):
                    self.terminal_states.append(state)
                
                # Track knock and gin actions
                action = state.get('action', '')
                if action == 'knock':
                    self.knock_states.append(state)
                elif action == 'gin':
                    self.gin_states.append(state)
            
            print(f"Loaded {len(self.games)} games with {len(states)} total states")
            print(f"Special states: {len(self.terminal_states)} terminal, {len(self.knock_states)} knock, {len(self.gin_states)} gin")
    
    def _create_hand_matrix(self, cards: List[int]) -> np.ndarray:
        """Convert list of card indices to 4x13 matrix."""
        matrix = np.zeros((4, 13), dtype=np.float32)
        for card_idx in cards:
            suit = card_idx // 13
            rank = card_idx % 13
            matrix[suit, rank] = 1
        return matrix
    
    def _create_discard_history(self, discards: List[int], max_len: int = 52) -> np.ndarray:
        """Convert discard pile to one-hot sequence."""
        history = np.zeros((max_len, 52), dtype=np.float32)
        for i, card_idx in enumerate(discards[-max_len:]):
            if card_idx >= 0 and i < max_len:  # Skip invalid indices
                history[i, card_idx] = 1
        return history
    
    def get_training_data(self, batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch of training data with emphasis on game-ending actions."""
        game_ids = list(self.games.keys())
        
        # Sample distribution: 
        # 30% knock states, 20% gin states, 20% terminal states, 30% random states
        knock_count = min(int(batch_size * 0.3), len(self.knock_states))
        gin_count = min(int(batch_size * 0.2), len(self.gin_states))
        terminal_count = min(int(batch_size * 0.2), len(self.terminal_states))
        random_count = batch_size - knock_count - gin_count - terminal_count
        
        # Initialize batch arrays
        hand_matrices = []
        discard_histories = []
        opponent_cards = []
        actions = []
        rewards = []
        dones = []
        
        # Sample knock states
        if knock_count > 0 and self.knock_states:
            for state in random.sample(self.knock_states, knock_count):
                self._process_state(state, hand_matrices, discard_histories, opponent_cards, actions, rewards, dones)
        
        # Sample gin states
        if gin_count > 0 and self.gin_states:
            for state in random.sample(self.gin_states, gin_count):
                self._process_state(state, hand_matrices, discard_histories, opponent_cards, actions, rewards, dones)
        
        # Sample terminal states
        if terminal_count > 0 and self.terminal_states:
            for state in random.sample(self.terminal_states, terminal_count):
                self._process_state(state, hand_matrices, discard_histories, opponent_cards, actions, rewards, dones)
        
        # Sample random states
        if random_count > 0:
            selected_games = np.random.choice(game_ids, min(random_count, len(game_ids)))
            for game_id in selected_games:
                # Randomly select a state from the game
                game_states = self.games[game_id]
                state_idx = np.random.randint(0, len(game_states))
                state = game_states[state_idx]
                self._process_state(state, hand_matrices, discard_histories, opponent_cards, actions, rewards, dones)
        
        # Convert to tensors
        batch = {
            'hand_matrix': torch.FloatTensor(np.array(hand_matrices))[:, None, :, :],  # Add channel dim
            'discard_history': torch.FloatTensor(np.array(discard_histories)),
            'opponent_model': torch.FloatTensor(np.array(opponent_cards)).reshape(len(hand_matrices), -1),
            'valid_actions_mask': torch.ones(len(hand_matrices), N_ACTIONS, dtype=torch.bool)  # All actions valid by default
        }
        
        return (
            batch,
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.BoolTensor(np.array(dones))
        )
    
    def _process_state(self, state, hand_matrices, discard_histories, opponent_cards, actions, rewards, dones):
        """Process a single state for training data."""
        # Create hand matrix
        hand_matrix = self._create_hand_matrix(state['playerHand'])
        hand_matrices.append(hand_matrix)
        
        # Create discard history
        discard_history = self._create_discard_history(state['discardPile'])
        discard_histories.append(discard_history)
        
        # Create opponent known cards
        opponent_matrix = self._create_hand_matrix(state.get('knownOpponentCards', []))
        opponent_cards.append(opponent_matrix)
        
        # Parse action string to get numeric index
        action_vec = np.zeros(N_ACTIONS, dtype=np.float32)
        if state['action'].startswith('draw_faceup_'):
            # Convert draw_faceup_True/False to action indices 0/1
            action_idx = 0 if state['action'].endswith('True') else 1
        elif state['action'].startswith('discard_'):
            # Convert discard_X to action index 2+X
            try:
                card_id = int(state['action'].split('_')[1])
                action_idx = 2 + card_id
            except:
                action_idx = -1
        elif state['action'] == 'knock':
            action_idx = 108  # Special action for knocking
        elif state['action'] == 'gin':
            action_idx = 109  # Special action for gin
        else:
            action_idx = -1  # Invalid/terminal state
        
        if action_idx >= 0 and action_idx < N_ACTIONS:
            action_vec[action_idx] = 1
        actions.append(action_vec)
        
        # Get reward and done flag
        # Boost rewards for winning actions to prioritize them
        reward = state.get('reward', 0)
        
        # Increase rewards for knock and gin actions to encourage the model to learn these
        if action_idx == 108:  # knock action
            # Reward is boosted if this is a winning action
            if reward > 0:
                reward *= 3.0  # Triple the reward for winning knocks
        elif action_idx == 109:  # gin action
            # Gin is the optimal winning action, give it a higher boost
            if reward > 0:
                reward *= 5.0  # 5x reward for winning gins
        
        rewards.append(reward)
        dones.append(state.get('isTerminal', False))
        
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        total_states = sum(len(states) for states in self.games.values())
        total_games = len(self.games)
        return {
            'total_games': total_games,
            'total_states': total_states,
            'terminal_states': len(self.terminal_states),
            'knock_states': len(self.knock_states),
            'gin_states': len(self.gin_states),
            'avg_game_length': total_states / max(1, total_games)
        }

class PrioritizedReplayBuffer:
    """Prioritized replay buffer for storing experiences."""
    def __init__(self, buffer_size: int):
        self.buffer = []
        self.buffer_size = buffer_size
        self.priorities = []
        self.position = 0
    
    def push(self, state: Dict[str, torch.Tensor], action: int, reward: float, 
            next_state: Dict[str, torch.Tensor], done: bool, priority: float = 1.0):
        """Add experience to buffer with priority."""
        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.buffer_size
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences based on priorities."""
        if len(self.buffer) == 0:
            return []
            
        # Convert priorities to probabilities
        probs = np.array(self.priorities) / sum(self.priorities)
        
        # Sample indices based on priorities
        indices = np.random.choice(len(self.buffer), 
                                 min(batch_size, len(self.buffer)), 
                                 p=probs,
                                 replace=False)
        
        return [self.buffer[idx] for idx in indices]
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for experiences."""
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = max(priority, 0.1)  # Ensure minimum priority
    
    def __len__(self) -> int:
        return len(self.buffer)

def create_training_dataset(input_files, output_file, sample_ratio=0.20):
    """Create a dataset with focus on game-ending states."""
    print(f"Creating training dataset with {sample_ratio*100}% of data...")
    
    all_states = []
    all_games = {}
    terminal_states = []
    
    # Load data from multiple files if provided
    if isinstance(input_files, str):
        input_files = [input_files]
    
    for input_file in input_files:
        print(f"Processing {input_file}...")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Group by game ID and collect terminal states
        for state in data:
            game_id = state.get('gameId', 0)
            if game_id not in all_games:
                all_games[game_id] = []
            all_games[game_id].append(state)
            
            # Track terminal states
            if state.get('isTerminal', False):
                terminal_states.append(state)
    
    # Sample games
    game_ids = list(all_games.keys())
    sample_size = max(1, int(len(game_ids) * sample_ratio))
    sampled_ids = random.sample(game_ids, sample_size)
    
    sampled_data = []
    for game_id in sampled_ids:
        sampled_data.extend(all_games[game_id])
    
    # Add extra terminal states to emphasize game endings
    # Add at most 20% extra terminal states to avoid overwhelming the dataset
    extra_terminal_count = min(len(terminal_states), int(len(sampled_data) * 0.2))
    if extra_terminal_count > 0:
        extra_terminals = random.sample(terminal_states, extra_terminal_count)
        sampled_data.extend(extra_terminals)
        print(f"Added {extra_terminal_count} additional terminal states to emphasize game endings")
    
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f)
    
    print(f"Created training dataset with {len(sampled_ids)} games ({len(sampled_data)} states)")
    return output_file

class GinRummyPolicyNetwork(nn.Module):
    """Policy network for REINFORCE with emphasis on winning actions."""
    def __init__(self, hidden_size: int = 192):
        super(GinRummyPolicyNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        
        # LSTM for history
        self.lstm = nn.LSTM(52, hidden_size, num_layers=2, batch_first=True, dropout=0.1)
        
        # Opponent model processing
        self.opponent_fc = nn.Linear(52, hidden_size // 2)
        
        # Action head
        conv_out_size = 32 * 4 * 13
        self.fc1 = nn.Linear(conv_out_size + hidden_size + hidden_size // 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.action_head = nn.Linear(hidden_size // 2, N_ACTIONS)
        
        # Temperature parameter for softmax (learnable)
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)
        
        # Action bias with stronger initialization
        self.register_parameter('action_bias', nn.Parameter(torch.zeros(N_ACTIONS)))
        with torch.no_grad():
            self.action_bias[108] = 2.0  # Knock
            self.action_bias[109] = 3.0  # Gin
        
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, hand_matrix, discard_history, opponent_model, valid_actions_mask):
        # Process hand
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(hand_matrix.size(0), -1)
        
        # Process discard history
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]
        
        # Process opponent model
        x3 = F.relu(self.opponent_fc(opponent_model))
        
        # Combine features
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        logits = self.action_head(x)
        
        # Add action bias with scaling
        logits = logits + self.action_bias * 5.0
        
        # Mask invalid actions
        logits = torch.where(valid_actions_mask, logits, torch.tensor(float('-inf')).to(logits.device))
        
        # Return both logits and probabilities
        temperature = F.softplus(self.temperature)  # Ensure temperature is positive
        probs = F.softmax(logits / temperature, dim=1)
        
        return logits, probs

class GinRummyREINFORCEAgent:
    """REINFORCE agent with emphasis on winning actions."""
    def __init__(self, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
            
        # Initialize network
        self.policy = GinRummyPolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Use replay buffer for experience collection
        self.memory = PrioritizedReplayBuffer(BUFFER_SIZE)
        
        self.steps_done = 0
        self.current_loss = 0
        
    def update(self, state_batch, action_batch, reward_batch):
        """Update policy network using REINFORCE with weighted actions and improved numerical stability."""
        # Move tensors to device if needed
        for key in state_batch:
            if not state_batch[key].device == self.device:
                state_batch[key] = state_batch[key].to(self.device)
                
        if not action_batch.device == self.device:
            action_batch = action_batch.to(self.device)
            
        if not reward_batch.device == self.device:
            reward_batch = reward_batch.to(self.device)
            
        # Forward pass through policy network
        logits, probs = self.policy(
            state_batch['hand_matrix'],
            state_batch['discard_history'],
            state_batch['opponent_model'],
            state_batch['valid_actions_mask']
        )
        
        # Calculate loss with emphasis on important actions
        action_indices = action_batch.argmax(dim=1)
        
        # Create weights to emphasize knock and gin actions
        action_weights = torch.ones_like(reward_batch)
        knock_mask = (action_indices == 108)
        gin_mask = (action_indices == 109)
        action_weights[knock_mask] = 2.0  # Double importance for knock
        action_weights[gin_mask] = 3.0    # Triple importance for gin
        
        # Normalize rewards for stability (using a softer normalization)
        reward_mean = reward_batch.mean()
        reward_std = reward_batch.std()
        if reward_std > 0:
            reward_batch = (reward_batch - reward_mean) / (reward_std + 1e-3)
        
        # Standard REINFORCE loss with importance sampling and improved numerical stability
        log_probs = F.log_softmax(logits, dim=1)
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        
        # Use softer clipping for log probabilities
        selected_log_probs = torch.clamp(selected_log_probs, min=-20.0, max=0.0)
        
        # Calculate policy loss with reward scaling and action weights
        policy_loss = (selected_log_probs * reward_batch * action_weights).mean()
        
        # Add entropy regularization for exploration (with softer clipping)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        entropy = torch.clamp(entropy, min=-5.0, max=5.0)  # Softer entropy clipping
        
        # Combine losses with scaled entropy term and stronger entropy coefficient
        loss = -policy_loss + 0.05 * entropy  # Note: negative sign is correct here
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        self.current_loss = loss.item()
        self.steps_done += 1
        
        return loss.item()
        
    def save(self, filepath):
        """Save model checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'steps_done': self.steps_done
        }
        torch.save(checkpoint, filepath)

def train_dqn(dataset_file, epochs=5, batch_size=1024, iterations_per_epoch=150):
    """Train DQN model with emphasis on winning actions."""
    print("Starting DQN training...")
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
    agent = GinRummyDQNAgent()
    
    # Track best loss for early stopping
    best_loss = float('inf')
    early_stop_patience = 3
    no_improvement_count = 0
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        total_batches = 0
        
        progress_bar = tqdm(range(iterations_per_epoch), desc="Training")
        for _ in progress_bar:
            # Get batch of training data
            state_batch, action_batch, reward_batch, done_batch = dataset.get_training_data(batch_size)
            
            # Scale up rewards for winning actions
            action_indices = torch.argmax(action_batch, dim=1)
            win_mask = (reward_batch > 0) & ((action_indices == 108) | (action_indices == 109))
            reward_batch[win_mask] *= 5.0  # Increased reward scaling
            
            # Add experiences to replay buffer with priority
            for i in range(min(128, batch_size)):
                priority = 1.0
                if action_indices[i] in [108, 109]:  # Knock or Gin actions
                    priority = 2.0
                if reward_batch[i] > 0:  # Winning moves
                    priority *= 2.0
                
                state = {
                    'hand_matrix': state_batch['hand_matrix'][i:i+1],
                    'discard_history': state_batch['discard_history'][i:i+1],
                    'valid_actions_mask': state_batch['valid_actions_mask'][i:i+1]
                }
                
                next_state = {
                    'hand_matrix': state_batch['hand_matrix'][i:i+1],
                    'discard_history': state_batch['discard_history'][i:i+1],
                    'valid_actions_mask': state_batch['valid_actions_mask'][i:i+1]
                }
                
                agent.memory.push(
                    state,
                    action_indices[i].item(),
                    reward_batch[i].item(),
                    next_state,
                    done_batch[i].item(),
                    priority
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
        
        # Save checkpoint
        model_path = f"models/dqn_enhanced_epoch_{epoch + 1}.pt"
        agent.save(model_path)
        print(f"Saved model checkpoint to {model_path}")
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improvement_count = 0
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
    print(f"Training complete! Total time: {(time.time() - start_time) / 60:.2f} minutes")
    return agent

def train_reinforce(dataset_file, epochs=5, batch_size=1024, iterations_per_epoch=150):
    """Train REINFORCE model with enhanced reward scaling and experience replay."""
    print("Starting REINFORCE training...")
    start_time = time.time()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = GinRummyDataset(dataset_file)
    stats = dataset.get_stats()
    print(f"Dataset stats: {stats}")
    
    # Initialize agent
    agent = GinRummyREINFORCEAgent()
    
    # Track best loss for early stopping
    best_loss = float('inf')
    early_stop_patience = 3
    no_improvement_count = 0
    
    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        total_batches = 0
        
        progress_bar = tqdm(range(iterations_per_epoch), desc="Training")
        for _ in progress_bar:
            # Get batch of training data
            state_batch, action_batch, reward_batch, done_batch = dataset.get_training_data(batch_size)
            
            # Scale up rewards for winning actions
            action_indices = torch.argmax(action_batch, dim=1)
            win_mask = (reward_batch > 0) & ((action_indices == 108) | (action_indices == 109))
            reward_batch[win_mask] *= 5.0  # Increased reward scaling
            
            # Add experiences to replay buffer with priority
            for i in range(min(128, batch_size)):
                priority = 1.0
                if action_indices[i] in [108, 109]:  # Knock or Gin actions
                    priority = 2.0
                if reward_batch[i] > 0:  # Winning moves
                    priority *= 2.0
                
                state = {
                    'hand_matrix': state_batch['hand_matrix'][i:i+1],
                    'discard_history': state_batch['discard_history'][i:i+1],
                    'opponent_model': state_batch['opponent_model'][i:i+1],
                    'valid_actions_mask': state_batch['valid_actions_mask'][i:i+1]
                }
                
                agent.memory.push(
                    state,
                    action_indices[i].item(),
                    reward_batch[i].item(),
                    state,  # REINFORCE doesn't use next_state, so we use the same state
                    done_batch[i].item(),
                    priority
                )
            
            # Train on replay buffer if enough data
            if len(agent.memory) >= 128:
                experiences = agent.memory.sample(min(batch_size, len(agent.memory)))
                if experiences:
                    batch = Experience(*zip(*experiences))
                    
                    replay_state_batch = {
                        'hand_matrix': torch.cat([s['hand_matrix'] for s in batch.state]).to(device),
                        'discard_history': torch.cat([s['discard_history'] for s in batch.state]).to(device),
                        'opponent_model': torch.cat([s['opponent_model'] for s in batch.state]).to(device),
                        'valid_actions_mask': torch.cat([s['valid_actions_mask'] for s in batch.state]).to(device)
                    }
                    
                    action_vecs = torch.zeros(len(batch.action), N_ACTIONS, device=device)
                    for j, a in enumerate(batch.action):
                        action_vecs[j, a] = 1
                    
                    replay_reward_batch = torch.tensor(batch.reward, device=device)
                    
                    loss = agent.update(replay_state_batch, action_vecs, replay_reward_batch)
                else:
                    loss = agent.update(state_batch, action_batch, reward_batch)
            else:
                loss = agent.update(state_batch, action_batch, reward_batch)
            
            # Track loss
            epoch_loss += loss
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss,
                'memory': len(agent.memory),
                'time': f"{(time.time() - start_time) / 60:.1f}m"
            })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / max(1, total_batches)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.6f}")
        
        # Save checkpoint
        model_path = f"models/reinforce_enhanced_epoch_{epoch + 1}.pt"
        agent.save(model_path)
        print(f"Saved model checkpoint to {model_path}")
        
        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            no_improvement_count = 0
            best_model_path = "models/reinforce_enhanced_best.pt"
            agent.save(best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            no_improvement_count += 1
            print(f"No improvement for {no_improvement_count} epochs")
            
            if no_improvement_count >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Save final model
    agent.save("models/reinforce_enhanced_final.pt")
    print(f"Training complete! Total time: {(time.time() - start_time) / 60:.2f} minutes")
    return agent

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Retrain Gin Rummy models with focus on winning actions")
    parser.add_argument("--model", choices=["dqn", "reinforce", "both"], default="both",
                        help="Which model type to train (default: both)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for training (default: 1024)")
    parser.add_argument("--iterations", type=int, default=150,
                        help="Iterations per epoch (default: 150)")
    parser.add_argument("--sample-ratio", type=float, default=0.15,
                        help="Ratio of data to sample for training (default: 0.15)")
    parser.add_argument("--data-dir", type=str, default="../java/MavenProject",
                        help="Directory containing training data (default: ../java/MavenProject)")
    
    args = parser.parse_args()
    
    print("Setting up training environment...")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Create training dataset
    input_files = []
    for i in range(1, 4):  # Use first 3 consolidated files
        file_path = os.path.join(args.data_dir, f"training_data_consolidated_{i}.json")
        if os.path.exists(file_path):
            input_files.append(file_path)
    
    if not input_files:
        print("No training data files found! Please check path.")
        return
        
    dataset_file = os.path.join(args.data_dir, "retrain_data.json")
    dataset_file = create_training_dataset(input_files, dataset_file, sample_ratio=args.sample_ratio)
    
    print(f"Dataset prepared. Training with {args.epochs} epochs, {args.iterations} iterations/epoch, batch size {args.batch_size}")
    
    # Train selected models
    if args.model in ["dqn", "both"]:
        print("\n=== TRAINING DQN MODEL ===")
        train_dqn(dataset_file, epochs=args.epochs, batch_size=args.batch_size, iterations_per_epoch=args.iterations)
    
    if args.model in ["reinforce", "both"]:
        print("\n=== TRAINING REINFORCE MODEL ===")
        train_reinforce(dataset_file, epochs=args.epochs, batch_size=args.batch_size, iterations_per_epoch=args.iterations)
    
    print("\n=== TRAINING COMPLETE ===")
    print("Retrained models are available in the models/ directory:")
    if args.model in ["dqn", "both"]:
        print("  - DQN: models/dqn_enhanced_final.pt")
        print("  - Best DQN: models/dqn_enhanced_best.pt")
    if args.model in ["reinforce", "both"]:
        print("  - REINFORCE: models/reinforce_enhanced_final.pt")
        print("  - Best REINFORCE: models/reinforce_enhanced_best.pt")

if __name__ == "__main__":
    main() 