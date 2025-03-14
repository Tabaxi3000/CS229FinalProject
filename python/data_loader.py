import json
import numpy as np
import torch
from typing import Dict, List, Tuple
from collections import defaultdict

class GinRummyDataset:
    def __init__(self, json_file: str):
        self.json_file = json_file
        self.games = defaultdict(list)  # gameId -> list of states
        self.load_data()
        
    def load_data(self):
        """Load and preprocess the JSON data."""
        print("Loading data from", self.json_file)
        with open(self.json_file, 'r') as f:
            data = json.load(f)
            
            # Extract game states array, handling both list and object formats
            if isinstance(data, dict):
                states = data.get('gameStates', [])
            else:  # data is already a list
                states = data
            
            # Group states by game
            game_ids = set()
            total_states = len(states)
            total_rewards = 0
            
            # Track current game being built
            current_game_id = None
            current_game_states = []
            
            for state in states:
                game_id = state.get('gameId', 0)
                
                # If we see a new game ID, save the previous game and start a new one
                if game_id != current_game_id:
                    if current_game_states:
                        self.games[current_game_id] = current_game_states
                        game_ids.add(current_game_id)
                    current_game_id = game_id
                    current_game_states = []
                
                current_game_states.append(state)
                total_rewards += state['reward']
            
            # Save the last game
            if current_game_states:
                self.games[current_game_id] = current_game_states
                game_ids.add(current_game_id)
            
            total_games = len(game_ids)
            avg_states_per_game = total_states / total_games if total_games > 0 else 0
            avg_reward = total_rewards / total_states if total_states > 0 else 0
            
            print(f"\nDataset Statistics:")
            print(f"Total games: {total_games}")
            print(f"Total states: {total_states}")
            print(f"Average states per game: {avg_states_per_game:.1f}")
            print(f"Average reward: {avg_reward:.3f}")
            
            # Print detailed stats about the first game
            if total_games > 0:
                first_game_id = min(game_ids)
                first_game = self.games[first_game_id]
                print(f"\nFirst Game Details:")
                print(f"Game ID: {first_game_id}")
                print(f"Number of states: {len(first_game)}")
                print(f"First state: {first_game[0]}")
                print(f"Last state: {first_game[-1]}")
                
                # Count action types
                action_counts = {}
                for state in first_game:
                    action = state['action']
                    action_counts[action] = action_counts.get(action, 0) + 1
                print(f"Action distribution: {action_counts}")
            
            if total_games < 900:  # Warn if we have significantly fewer games than expected
                print(f"\nWarning: Only loaded {total_games} games, expected around 1000 per file")
                # Try to debug the data structure
                if isinstance(data, dict):
                    print("Data keys:", data.keys())
                print("First few states:", states[:2])
        
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
            if card_idx >= 0:  # Skip invalid indices
                history[i, card_idx] = 1
        return history
    
    def get_training_data(self, batch_size: int = 32) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get batch of training data in format needed by models."""
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
                card_id = int(state['action'].split('_')[1])
                action_idx = 2 + card_id
            elif state['action'] == 'knock':
                action_idx = 108  # Special action for knocking
            elif state['action'] == 'gin':
                action_idx = 109  # Special action for gin
            else:
                action_idx = -1  # Invalid/terminal state
            
            if action_idx >= 0:
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
    
    def get_episode_trajectories(self) -> List[List[Dict]]:
        """Get complete episode trajectories for algorithms that need them."""
        return list(self.games.values())
    
    def get_stats(self) -> Dict:
        """Get dataset statistics."""
        total_states = sum(len(states) for states in self.games.values())
        total_games = len(self.games)
        avg_game_length = total_states / total_games
        
        action_counts = defaultdict(int)
        reward_sum = 0
        terminal_states = 0
        
        for states in self.games.values():
            for state in states:
                # Parse action string to get action type
                action = state['action']
                if action.startswith('draw_faceup_'):
                    action_type = 'draw_faceup'
                elif action.startswith('discard_'):
                    action_type = 'discard'
                elif action in ['knock', 'gin', 'game_over']:
                    action_type = action
                else:
                    action_type = 'other'
                    
                action_counts[action_type] += 1
                reward_sum += state['reward']
                if state['isTerminal']:
                    terminal_states += 1
        
        return {
            'total_games': total_games,
            'total_states': total_states,
            'avg_game_length': avg_game_length,
            'action_distribution': dict(action_counts),
            'avg_reward': reward_sum / total_states,
            'terminal_states': terminal_states
        } 