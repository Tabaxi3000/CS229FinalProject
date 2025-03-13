import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
from collections import namedtuple

# Constants
N_SUITS = 4
N_RANKS = 13
N_ACTIONS = 110  # 52 discards + 52 draws + 6 special actions
GAMMA = 0.99  # Discount factor
ENTROPY_COEF = 0.01  # Entropy regularization coefficient

Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size: int = 256):
        super(PolicyNetwork, self).__init__()
        
        # Process hand matrix (4x13)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Process discard history
        self.lstm = nn.LSTM(52, hidden_size, batch_first=True)
        
        # Process opponent model
        self.opponent_fc = nn.Linear(52, hidden_size)
        
        # Combine all features
        self.fc1 = nn.Linear(64 * 4 * 13 + hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, N_ACTIONS)
        
    def forward(self, hand_matrix, discard_history, opponent_model, valid_actions_mask):
        # Process hand
        x1 = F.relu(self.conv1(hand_matrix))
        x1 = F.relu(self.conv2(x1))
        x1 = x1.view(-1, 64 * 4 * 13)
        
        # Process discard history
        x2, _ = self.lstm(discard_history)
        x2 = x2[:, -1, :]  # Take last LSTM output
        
        # Process opponent model
        x3 = F.relu(self.opponent_fc(opponent_model))
        
        # Combine features
        x = torch.cat([x1, x2, x3], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Get action logits and apply mask
        action_logits = self.action_head(x)
        action_logits[~valid_actions_mask] = float('-inf')
        
        # Get action probabilities
        action_probs = F.softmax(action_logits, dim=1)
        
        return action_probs

class REINFORCEAgent:
    def __init__(self, learning_rate: float = 0.001):
        self.policy = PolicyNetwork()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.saved_experiences = []
        
    def select_action(self, state: Dict[str, torch.Tensor], valid_actions_mask: torch.Tensor) -> int:
        """Select action using current policy."""
        with torch.no_grad():
            probs = self.policy(
                state['hand_matrix'],
                state['discard_history'],
                state['opponent_model'],
                valid_actions_mask
            )
        
        # Sample action from probability distribution
        action = torch.multinomial(probs, 1).item()
        return action
    
    def store_experience(self, *, state: Dict[str, torch.Tensor], action: int, reward: float, next_state: Optional[Dict[str, torch.Tensor]], done: bool):
        """Store experience for later training. Uses keyword arguments for clarity."""
        experience = Experience(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self.saved_experiences.append(experience)
    
    def calculate_returns(self, rewards: List[float]) -> torch.Tensor:
        """Calculate discounted returns for each timestep."""
        returns = []
        R = 0
        
        for r in reversed(rewards):
            R = r + GAMMA * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        return returns
    
    def calculate_intermediate_reward(self, state: Dict[str, torch.Tensor]) -> float:
        """Calculate intermediate rewards based on game state."""
        reward = 0.0
        
        # Reward for meld formation (+0.1 per card in valid set/run)
        melds = self._find_melds(state['hand_matrix'])
        reward += 0.1 * sum(len(meld) for meld in melds)
        
        # Penalty for deadwood (-0.05 per unmelded point)
        deadwood = self._calculate_deadwood(state['hand_matrix'], melds)
        reward -= 0.05 * deadwood
        
        # Bonus for successful knocks and gin
        if state.get('knocked', False):
            reward += 1.0
        if state.get('gin', False):
            reward += 2.0
            
        return reward
    
    def train(self):
        """Train policy using collected experiences."""
        if not self.saved_experiences:
            return
            
        states = [exp.state for exp in self.saved_experiences]
        actions = torch.tensor([exp.action for exp in self.saved_experiences])
        rewards = [exp.reward for exp in self.saved_experiences]
        
        # Calculate returns
        returns = self.calculate_returns(rewards)
        
        # Calculate loss
        loss = 0
        for t, (state, action, R) in enumerate(zip(states, actions, returns)):
            # Get action probabilities
            probs = self.policy(
                state['hand_matrix'],
                state['discard_history'],
                state['opponent_model'],
                state['valid_actions_mask']
            )
            
            # Calculate policy gradient loss
            log_prob = torch.log(probs[0, action])  # Add batch dimension indexing
            policy_loss = -log_prob * R
            
            # Add entropy regularization
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            loss += policy_loss - ENTROPY_COEF * entropy
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Clear experiences
        self.saved_experiences = []
    
    def _find_melds(self, hand_matrix: torch.Tensor) -> List[List[int]]:
        """Find all valid melds in the hand."""
        # Convert hand matrix to card indices
        hand = []
        for suit in range(4):
            for rank in range(13):
                if hand_matrix[0, suit, rank] == 1:
                    hand.append(suit * 13 + rank)
        
        melds = []
        # Check for sets (same rank, different suits)
        for rank in range(13):
            same_rank = [card for card in hand if card % 13 == rank]
            if len(same_rank) >= 3:
                # Add all possible combinations of 3 or 4 cards
                if len(same_rank) >= 4:
                    melds.append(same_rank)  # All 4 cards
                for i in range(len(same_rank)):
                    for j in range(i + 1, len(same_rank)):
                        for k in range(j + 1, len(same_rank)):
                            melds.append([same_rank[i], same_rank[j], same_rank[k]])
        
        # Check for runs (same suit, consecutive ranks)
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
        # Convert hand matrix to card indices
        hand = []
        for suit in range(4):
            for rank in range(13):
                if hand_matrix[0, suit, rank] == 1:
                    hand.append(suit * 13 + rank)
        
        # Remove melded cards
        unmelded = set(hand)
        for meld in melds:
            for card in meld:
                unmelded.discard(card)
        
        # Calculate deadwood points
        deadwood = 0
        for card in unmelded:
            rank = card % 13
            # Face cards (11-13) are worth 10 points
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
        """Convert discard pile to one-hot sequence."""
        history = np.zeros((max_len, 52), dtype=np.float32)
        for i, card_idx in enumerate(discards[-max_len:]):
            if card_idx >= 0:  # Skip invalid indices
                history[i, card_idx] = 1
        return history[None, ...]  # Add batch dimension
    
    def _create_opponent_model(self, cards: List[int]) -> np.ndarray:
        """Convert list of known opponent cards to probability vector."""
        model = np.zeros(52, dtype=np.float32)
        for card_idx in cards:
            model[card_idx] = 1
        return model[None, ...]  # Add batch dimension
    
    def update(self, state_batch, action_batch, reward_batch):
        """Perform a REINFORCE update using batched data."""
        # Extract batch size
        batch_size = state_batch['hand_matrix'].size(0)
        
        # Forward pass through policy network
        log_probs = self.policy(
            state_batch['hand_matrix'],
            state_batch['discard_history'],
            state_batch['opponent_model'],
            state_batch['valid_actions_mask']
        )
        
        # Calculate loss
        action_indices = action_batch.argmax(dim=1)
        selected_log_probs = log_probs.gather(1, action_indices.unsqueeze(1)).squeeze(1)
        loss = -(selected_log_probs * reward_batch).mean()
        
        # Add entropy regularization
        probs = F.softmax(log_probs, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss -= ENTROPY_COEF * entropy
        
        # Backpropagate
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
        
    def load(self, filepath):
        """Load model checkpoint from file."""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 