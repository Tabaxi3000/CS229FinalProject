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
BATCH_SIZE = 1024  # Increased batch size
BUFFER_SIZE = 10000  # Smaller buffer for faster filling
GAMMA = 0.99
ENTROPY_COEF = 0.01

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

# ----------------- SIMPLIFIED MODEL ARCHITECTURES -----------------

class SimpleDQNetwork(nn.Module):
    """Simplified DQN architecture with fewer parameters for faster training"""
    def __init__(self, hidden_size: int = 128):  # Reduced hidden size
        super(SimpleDQNetwork, self).__init__()
        
        # Simplified convolutional layers
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Single conv layer with fewer filters
        
        # Simplified LSTM layer
        self.lstm = nn.LSTM(52, hidden_size, num_layers=1, batch_first=True)  # Single layer LSTM
        
        # Simplified fully connected layers
        conv_out_size = 16 * 4 * 13
        self.fc1 = nn.Linear(conv_out_size + hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, N_ACTIONS)
        
    def forward(self, hand_matrix, discard_history, valid_actions_mask):
        # Process hand matrix through simplified conv layer
        x1 = F.relu(self.conv(hand_matrix))
        x1 = x1.view(hand_matrix.size(0), -1)  # Flatten
        
        # Process discard history through LSTM
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take the last output from the LSTM
        x2 = x2.view(x2.size(0), -1)  # Flatten the LSTM output for concatenation
        
        # Combine features
        x = torch.cat([x1, x2], dim=1)
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        # Mask invalid actions
        q_values = torch.where(valid_actions_mask, q_values, torch.tensor(float('-inf')).to(q_values.device))
        
        return q_values

class SimplePolicyNetwork(nn.Module):
    """Simplified Policy Network for REINFORCE with fewer parameters"""
    def __init__(self, hidden_size: int = 128):  # Reduced hidden size
        super(SimplePolicyNetwork, self).__init__()
        
        # Simplified hand processing
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)  # Single conv layer
        
        # Simplified history processing
        self.lstm = nn.LSTM(52, hidden_size, num_layers=1, batch_first=True)  # Single layer
        
        # Simplified opponent model processing
        self.opponent_fc = nn.Linear(52, hidden_size)
        
        # Simplified action head
        self.fc1 = nn.Linear(16 * 4 * 13 + hidden_size * 2, hidden_size)
        self.action_head = nn.Linear(hidden_size, N_ACTIONS)
        
    def forward(self, hand_matrix, discard_history, opponent_model, valid_actions_mask, temperature=1.0):
        # Process hand
        x1 = F.relu(self.conv(hand_matrix))
        x1 = x1.view(-1, 16 * 4 * 13)
        
        # Process discard history
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take the last output from the LSTM
        x2 = x2.view(x2.size(0), -1)  # Flatten the LSTM output for concatenation
        
        # Process opponent model
        x3 = F.relu(self.opponent_fc(opponent_model))
        
        # Combine features
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        action_logits = self.action_head(x)
        
        # Apply stronger masking for invalid actions
        # Use a large negative value instead of -inf to avoid numerical issues
        invalid_mask = ~valid_actions_mask
        action_logits = action_logits.masked_fill(invalid_mask, -1e9)
        
        # Apply temperature scaling for exploration/exploitation control
        # Lower temperature makes the distribution more peaked (more exploitation)
        # Higher temperature makes the distribution more uniform (more exploration)
        if temperature != 1.0:
            action_logits = action_logits / temperature
        
        # Get action probabilities with improved numerical stability
        action_probs = F.softmax(action_logits, dim=1)
        
        # Ensure zero probability for invalid actions
        action_probs = action_probs.masked_fill(invalid_mask, 0.0)
        
        # Renormalize probabilities to sum to 1
        action_probs = action_probs / (action_probs.sum(dim=1, keepdim=True) + 1e-10)
        
        return action_probs

# ----------------- SIMPLIFIED AGENTS -----------------

class ReplayBuffer:
    """Simplified Replay Buffer with smaller capacity"""
    def __init__(self, buffer_size: int):
        self.buffer = deque(maxlen=buffer_size)
    
    def push(self, *, state: Dict[str, torch.Tensor], action: int, reward: float, next_state: Dict[str, torch.Tensor], done: bool):
        """Add experience to buffer using named arguments."""
        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample random batch of experiences."""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def __len__(self) -> int:
        return len(self.buffer)

class FastDQNAgent:
    """Simplified DQN agent for faster training"""
    def __init__(self, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
        
        # Initialize simplified networks
        self.policy_net = SimpleDQNetwork().to(self.device)
        self.target_net = SimpleDQNetwork().to(self.device)
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
        if len(self.memory) < 64:  # Lower threshold for faster training
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
            self.optimizer.step()
        
        self.current_loss = loss.item()
        
        # Update target network more frequently for faster convergence
        if self.steps_done % 100 == 0:  # More frequent updates
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

class FastREINFORCEAgent:
    """Simplified REINFORCE agent for faster training"""
    def __init__(self, learning_rate: float = 0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = torch.device("mps")  # For Apple Silicon
            
        # Initialize simplified network
        self.policy = SimplePolicyNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # For mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
    
    def update(self, state_batch, action_batch, reward_batch):
        """Perform a REINFORCE update with mixed precision if available"""
        # Move tensors to device if they aren't already
        for key in state_batch:
            if not state_batch[key].device == self.device:
                state_batch[key] = state_batch[key].to(self.device)
                
        if not action_batch.device == self.device:
            action_batch = action_batch.to(self.device)
            
        if not reward_batch.device == self.device:
            reward_batch = reward_batch.to(self.device)
        
        # Use mixed precision if available
        if self.scaler:
            with torch.cuda.amp.autocast():
                # Forward pass through policy network
                log_probs = self.policy(
                    state_batch['hand_matrix'],
                    state_batch['discard_history'],
                    state_batch['opponent_model'],
                    state_batch['valid_actions_mask']
                )
                
                # Calculate loss
                action_indices = action_batch.argmax(dim=1)
                selected_log_probs = torch.log(log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1) + 1e-10)
                loss = -(selected_log_probs * reward_batch).mean()
                
                # Add entropy regularization
                entropy = -torch.sum(log_probs * torch.log(log_probs + 1e-10), dim=1).mean()
                loss -= ENTROPY_COEF * entropy
            
            # Optimize with mixed precision
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # Standard computation without mixed precision
            # Forward pass through policy network
            log_probs = self.policy(
                state_batch['hand_matrix'],
                state_batch['discard_history'],
                state_batch['opponent_model'],
                state_batch['valid_actions_mask']
            )
            
            # Calculate loss
            action_indices = action_batch.argmax(dim=1)
            selected_log_probs = torch.log(log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1) + 1e-10)
            loss = -(selected_log_probs * reward_batch).mean()
            
            # Add entropy regularization
            entropy = -torch.sum(log_probs * torch.log(log_probs + 1e-10), dim=1).mean()
            loss -= ENTROPY_COEF * entropy
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return loss.item()
    
    def save(self, filepath):
        """Save model checkpoint to file."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        torch.save(checkpoint, filepath)

# ----------------- DATA HANDLING -----------------

class GinRummyDataset:
    """Simplified dataset class that loads directly from a file"""
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.games = {}  # gameId -> list of states
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the JSON data."""
        print(f"Loading data from {self.json_file}")
        with open(self.json_file, 'r') as f:
            data = json.load(f)
            
            # Extract game states array
            if isinstance(data, dict):
                states = data.get('gameStates', [])
            else:  # data is already a list
                states = data
            
            # Group states by game
            for state in states:
                game_id = state.get('gameId', 0)
                if game_id not in self.games:
                    self.games[game_id] = []
                self.games[game_id].append(state)
            
            print(f"Loaded {len(self.games)} games with {len(states)} total states")
    
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
        """Get batch of training data."""
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
            action_vec = np.zeros(110, dtype=np.float32)  # 110 possible actions
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
            
            if action_idx >= 0 and action_idx < 110:
                action_vec[action_idx] = 1
            actions.append(action_vec)
            
            # Get reward and done flag
            rewards.append(state['reward'])
            dones.append(state['isTerminal'])
        
        # Convert to tensors
        batch = {
            'hand_matrix': torch.FloatTensor(np.array(hand_matrices))[:, None, :, :],  # Add channel dim
            'discard_history': torch.FloatTensor(np.array(discard_histories)),
            'opponent_model': torch.FloatTensor(np.array(opponent_cards)).reshape(batch_size, -1),
            'valid_actions_mask': torch.ones(batch_size, 110, dtype=torch.bool)  # All actions valid by default
        }
        
        return (
            batch,
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)),
            torch.BoolTensor(np.array(dones))
        )
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        total_states = sum(len(states) for states in self.games.values())
        total_games = len(self.games)
        return {
            'total_games': total_games,
            'total_states': total_states,
            'avg_game_length': total_states / max(1, total_games)
        }

def create_mini_dataset(input_file, output_file, sample_ratio=0.05):
    """Create a tiny dataset with just 5% of the games."""
    print(f"Creating mini dataset with {sample_ratio*100}% of data...")
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Group by game ID
    games = {}
    for state in data:
        game_id = state.get('gameId', 0)
        if game_id not in games:
            games[game_id] = []
        games[game_id].append(state)
    
    # Sample a small subset of games
    game_ids = list(games.keys())
    sample_size = max(1, int(len(game_ids) * sample_ratio))
    sampled_ids = random.sample(game_ids, sample_size)
    
    sampled_data = []
    for game_id in sampled_ids:
        sampled_data.extend(games[game_id])
    
    with open(output_file, 'w') as f:
        json.dump(sampled_data, f)
    
    print(f"Created mini dataset with {len(sampled_ids)} games ({len(sampled_data)} states)")
    return output_file

# ----------------- TRAINING FUNCTIONS -----------------

def train_quick_dqn(mini_file, epochs=5, batch_size=1024, iterations_per_epoch=200):
    """Ultra-fast DQN training on mini dataset."""
    print("Starting ultra-fast DQN training...")
    start_time = time.time()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = GinRummyDataset(mini_file)
    stats = dataset.get_stats()
    print(f"Dataset stats: {stats}")
    
    # Initialize agent
    agent = FastDQNAgent()
    
    # Training loop with fixed number of iterations per epoch
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        total_batches = 0
        
        # Fixed number of iterations
        progress_bar = tqdm(range(iterations_per_epoch), desc="Training")
        for _ in progress_bar:
            # Get batch of training data
            state_batch, action_batch, reward_batch, done_batch = dataset.get_training_data(batch_size)
            
            # Add experiences to replay buffer (use just a subset to speed up)
            for i in range(min(64, batch_size)):  # Only use part of the batch for faster buffer filling
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
        
        # Save checkpoint after each epoch for ultra-fast training
        model_path = f"models/dqn_quick_epoch_{epoch + 1}.pt"
        agent.save(model_path)
        print(f"Saved model checkpoint to {model_path}")
    
    # Save final model
    agent.save("models/dqn_quick_final.pt")
    
    # Print training time
    training_time = (time.time() - start_time) / 60
    print(f"Quick DQN training complete! Final model saved. Total training time: {training_time:.2f} minutes")
    return agent

def train_quick_reinforce(mini_file, epochs=5, batch_size=1024, iterations_per_epoch=200):
    """Ultra-fast REINFORCE training on mini dataset."""
    print("Starting ultra-fast REINFORCE training...")
    start_time = time.time()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = GinRummyDataset(mini_file)
    stats = dataset.get_stats()
    print(f"Dataset stats: {stats}")
    
    # Initialize agent
    agent = FastREINFORCEAgent()
    
    # Training loop with fixed number of iterations per epoch
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        epoch_loss = 0.0
        total_batches = 0
        
        # Fixed number of iterations
        progress_bar = tqdm(range(iterations_per_epoch), desc="Training")
        for _ in progress_bar:
            # Get batch of training data
            state_batch, action_batch, reward_batch, _ = dataset.get_training_data(batch_size)
            
            # Perform REINFORCE update
            loss = agent.update(state_batch, action_batch, reward_batch)
            
            # Track loss
            epoch_loss += loss
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss,
                'time': f"{(time.time() - start_time) / 60:.1f}m"
            })
        
        # Calculate average epoch loss
        avg_epoch_loss = epoch_loss / max(1, total_batches)
        print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.6f}")
        
        # Save checkpoint after each epoch for ultra-fast training
        model_path = f"models/reinforce_quick_epoch_{epoch + 1}.pt"
        agent.save(model_path)
        print(f"Saved model checkpoint to {model_path}")
    
    # Save final model
    agent.save("models/reinforce_quick_final.pt")
    
    # Print training time
    training_time = (time.time() - start_time) / 60
    print(f"Quick REINFORCE training complete! Final model saved. Total training time: {training_time:.2f} minutes")
    return agent

def main():
    # Create mini dataset from first file
    first_file = "../java/MavenProject/training_data_consolidated_1.json"
    mini_file = "../java/MavenProject/mini_training_data.json"
    
    # Use a very small subset for ultra-fast training
    create_mini_dataset(first_file, mini_file, sample_ratio=0.05)  # Just 5% of the data
    
    # Train both models in sequence (REINFORCE first, then DQN)
    print("\n\n=== STARTING REINFORCE TRAINING (FASTER) ===\n")
    train_quick_reinforce(mini_file, epochs=5, batch_size=1024, iterations_per_epoch=200)
    
    print("\n\n=== STARTING DQN TRAINING ===\n")
    train_quick_dqn(mini_file, epochs=5, batch_size=1024, iterations_per_epoch=200)
    
    print("\n\n=== TRAINING COMPLETE ===\n")
    print("Preliminary models are available in the models/ directory")
    print("  - REINFORCE: models/reinforce_quick_final.pt")
    print("  - DQN: models/dqn_quick_final.pt")

if __name__ == "__main__":
    main() 