import os
import time
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import copy
import json

# Try importing the model files
try:
    from dqn import DQNAgent
    from reinforce import REINFORCEAgent
    from mcts import MCTSAgent, MCTSPolicyNetwork, MCTSValueNetwork
    from quick_train import FastDQNAgent, FastREINFORCEAgent
    from enhanced_train import EnhancedDQNAgent, EnhancedREINFORCEAgent
except ImportError as e:
    print(f"Warning: Could not import some model classes: {e}")

# Constants
NUM_GAMES = 100  # Number of games to play for each evaluation
MAX_TURNS = 100  # Maximum number of turns per game to prevent infinite loops
SUITS = ['H', 'D', 'C', 'S']  # Hearts, Diamonds, Clubs, Spades
RANKS = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
DEADWOOD_THRESHOLD = 10  # Threshold for gin/knock
VERBOSE = False  # Detailed output for debugging

# Card constants
TOTAL_CARDS = len(SUITS) * len(RANKS)
HAND_SIZE = 10

# Action constants
DRAW_STOCK = 0
DRAW_DISCARD = 1
DISCARD = 2  # Base action for discarding, add card index to get specific discard action
KNOCK = 3
GIN = 4
ACTION_OFFSET = 5  # Offset for card-specific actions (discard)

class GinRummyGame:
    """A simplified implementation of Gin Rummy for AI model evaluation."""
    
    def __init__(self, verbose=False):
        """Initialize a new game."""
        self.verbose = verbose
        self.reset()
    
    def reset(self):
        """Reset the game to a new state."""
        # Initialize deck
        self.deck = list(range(TOTAL_CARDS))
        random.shuffle(self.deck)
        
        # Deal hands
        self.player_hands = [[], []]
        for _ in range(HAND_SIZE):
            self.player_hands[0].append(self.deck.pop())
            self.player_hands[1].append(self.deck.pop())
        
        # Sort hands for easier viewing
        self.player_hands[0].sort()
        self.player_hands[1].sort()
        
        # Initialize discard pile with one card
        self.discard_pile = [self.deck.pop()]
        
        # Game state
        self.current_player = 0
        self.turn_count = 0
        self.drawn_card = None
        self.last_action = None
        self.game_over = False
        self.winner = None
        self.knock_player = None
        
        # Known opponent cards (for tracking what each player knows)
        self.known_opponent_cards = [[], []]
        
        if self.verbose:
            print("Game reset")
            print(f"Player 1 hand: {self._format_hand(self.player_hands[0])}")
            print(f"Player 2 hand: {self._format_hand(self.player_hands[1])}")
            print(f"Top discard: {self._format_card(self.discard_pile[-1])}")
        
        return self.get_state(0)
    
    def get_state(self, player_id):
        """Get the current state from a player's perspective."""
        # Basic state information
        state = {
            'playerHand': self.player_hands[player_id].copy(),
            'discardPile': self.discard_pile.copy(),
            'knownOpponentCards': self.known_opponent_cards[player_id].copy(),
            'drawnCard': self.drawn_card,
            'currentPlayer': self.current_player,
            'turnCount': self.turn_count,
            'gameOver': self.game_over,
            'winner': self.winner
        }
        
        # Add valid actions
        state['validActions'] = self.get_valid_actions(player_id)
        
        return state
    
    def get_valid_actions(self, player_id):
        """Get valid actions for the current player."""
        if self.game_over or player_id != self.current_player:
            return []
        
        valid_actions = []
        
        if self.drawn_card is None:
            # If no card drawn yet, player can draw from stock or discard
            valid_actions.append(DRAW_STOCK)
            if self.discard_pile:
                valid_actions.append(DRAW_DISCARD)
        else:
            # Player has drawn, must discard
            for card_idx in range(HAND_SIZE + 1):  # +1 for drawn card
                if card_idx < HAND_SIZE:
                    valid_actions.append(DISCARD + self.player_hands[player_id][card_idx])
                else:
                    valid_actions.append(DISCARD + self.drawn_card)
            
            # Check if player can knock or gin
            deadwood = self.calculate_deadwood(self.player_hands[player_id] + [self.drawn_card])
            if deadwood <= DEADWOOD_THRESHOLD:
                valid_actions.append(KNOCK)
                if deadwood == 0:
                    valid_actions.append(GIN)
        
        return valid_actions
    
    def step(self, player_id, action):
        """Take a step in the game based on player action."""
        if self.game_over or player_id != self.current_player:
            if self.verbose:
                print(f"Invalid action: game over or not player {player_id}'s turn")
            return self.get_state(player_id), -1, True
        
        reward = 0
        
        # Process action
        if action == DRAW_STOCK and self.drawn_card is None:
            # Draw from stock
            if self.deck:
                self.drawn_card = self.deck.pop()
                if self.verbose:
                    print(f"Player {player_id+1} draws from stock: {self._format_card(self.drawn_card)}")
            else:
                # Deck is empty, end in draw
                self.game_over = True
                self.winner = None
                if self.verbose:
                    print("Game ends in a draw - deck empty")
                return self.get_state(player_id), 0, True
                
        elif action == DRAW_DISCARD and self.drawn_card is None:
            # Draw from discard pile
            if self.discard_pile:
                self.drawn_card = self.discard_pile.pop()
                # Other player knows this card
                other_player = 1 - player_id
                self.known_opponent_cards[other_player].append(self.drawn_card)
                if self.verbose:
                    print(f"Player {player_id+1} draws from discard: {self._format_card(self.drawn_card)}")
            else:
                # Should not happen with proper valid action checking
                if self.verbose:
                    print("Invalid action: discard pile is empty")
                return self.get_state(player_id), -1, False
                
        elif DISCARD <= action < KNOCK and self.drawn_card is not None:
            # Discard a card
            card_to_discard = action - DISCARD
            
            # Check if card is valid to discard
            if card_to_discard == self.drawn_card:
                # Discard the drawn card
                self.discard_pile.append(self.drawn_card)
                if self.verbose:
                    print(f"Player {player_id+1} discards drawn card: {self._format_card(self.drawn_card)}")
                self.drawn_card = None
            elif card_to_discard in self.player_hands[player_id]:
                # Discard from hand and add drawn card to hand
                self.player_hands[player_id].remove(card_to_discard)
                self.player_hands[player_id].append(self.drawn_card)
                self.player_hands[player_id].sort()
                self.discard_pile.append(card_to_discard)
                if self.verbose:
                    print(f"Player {player_id+1} discards: {self._format_card(card_to_discard)}")
                self.drawn_card = None
            else:
                # Invalid card
                if self.verbose:
                    print(f"Invalid discard: card {card_to_discard} not in hand or drawn")
                return self.get_state(player_id), -1, False
            
            # End turn
            self.current_player = 1 - player_id
            self.turn_count += 1
            
            # Check for turn limit
            if self.turn_count >= MAX_TURNS:
                self.game_over = True
                self.winner = None  # Draw
                if self.verbose:
                    print("Game ends in a draw - turn limit reached")
                return self.get_state(player_id), 0, True
                
        elif action == KNOCK and self.drawn_card is not None:
            # Player knocks
            # Add drawn card to hand temporarily to check deadwood
            temp_hand = self.player_hands[player_id] + [self.drawn_card]
            deadwood = self.calculate_deadwood(temp_hand)
            
            if deadwood <= DEADWOOD_THRESHOLD:
                # Valid knock
                self.knock_player = player_id
                self.player_hands[player_id].append(self.drawn_card)
                self.player_hands[player_id].sort()
                self.drawn_card = None
                
                # Calculate opponent's deadwood
                other_player = 1 - player_id
                opponent_deadwood = self.calculate_deadwood(self.player_hands[other_player])
                
                # Determine winner
                if deadwood < opponent_deadwood:
                    # Knocker wins
                    self.winner = player_id
                    reward = 25 + (opponent_deadwood - deadwood)
                    if self.verbose:
                        print(f"Player {player_id+1} knocks and wins! Deadwood: {deadwood} vs {opponent_deadwood}")
                else:
                    # Opponent wins (undercut)
                    self.winner = other_player
                    reward = -25 - (opponent_deadwood - deadwood)
                    if self.verbose:
                        print(f"Player {player_id+1} knocks but is undercut! Deadwood: {deadwood} vs {opponent_deadwood}")
                
                self.game_over = True
                return self.get_state(player_id), reward, True
            else:
                # Invalid knock
                if self.verbose:
                    print(f"Invalid knock: deadwood {deadwood} > {DEADWOOD_THRESHOLD}")
                return self.get_state(player_id), -1, False
                
        elif action == GIN and self.drawn_card is not None:
            # Player declares gin
            # Add drawn card to hand temporarily to check deadwood
            temp_hand = self.player_hands[player_id] + [self.drawn_card]
            deadwood = self.calculate_deadwood(temp_hand)
            
            if deadwood == 0:
                # Valid gin
                self.player_hands[player_id].append(self.drawn_card)
                self.player_hands[player_id].sort()
                self.drawn_card = None
                
                # Gin bonus
                self.winner = player_id
                self.game_over = True
                
                # Calculate opponent's deadwood for scoring
                other_player = 1 - player_id
                opponent_deadwood = self.calculate_deadwood(self.player_hands[other_player])
                
                reward = 25 + opponent_deadwood
                if self.verbose:
                    print(f"Player {player_id+1} gets gin! Deadwood: 0 vs {opponent_deadwood}")
                
                return self.get_state(player_id), reward, True
            else:
                # Invalid gin
                if self.verbose:
                    print(f"Invalid gin: deadwood {deadwood} != 0")
                return self.get_state(player_id), -1, False
        else:
            # Invalid action
            if self.verbose:
                print(f"Invalid action: {action}")
            return self.get_state(player_id), -1, False
        
        # Return state, reward, done
        return self.get_state(player_id), reward, self.game_over
    
    def find_melds(self, hand):
        """Find all possible melds in a hand."""
        # Group cards by suit and rank
        suits = [[] for _ in range(len(SUITS))]
        ranks = [[] for _ in range(len(RANKS))]
        
        for card in hand:
            suit = card // len(RANKS)
            rank = card % len(RANKS)
            suits[suit].append(card)
            ranks[rank].append(card)
        
        melds = []
        
        # Find sets (same rank, different suits)
        for rank in range(len(RANKS)):
            if len(ranks[rank]) >= 3:
                melds.append(sorted(ranks[rank]))
        
        # Find runs (same suit, consecutive ranks)
        for suit in range(len(SUITS)):
            suit_cards = sorted([card % len(RANKS) for card in suits[suit]])
            run = []
            
            for i, rank in enumerate(suit_cards):
                if i > 0 and rank != suit_cards[i-1] + 1:
                    if len(run) >= 3:
                        melds.append([suit * len(RANKS) + r for r in run])
                    run = []
                run.append(rank)
            
            if len(run) >= 3:
                melds.append([suit * len(RANKS) + r for r in run])
        
        return melds
    
    def calculate_deadwood(self, hand):
        """Calculate the deadwood value of a hand."""
        melds = self.find_melds(hand)
        
        # If no melds, all cards are deadwood
        if not melds:
            return sum(min(DEADWOOD_THRESHOLD, card % len(RANKS) + 1) for card in hand)
        
        # Try different combinations of non-overlapping melds
        # This is a simplified approach, not the optimal solution
        best_deadwood = float('inf')
        
        # Just use the first meld for simplicity
        for meld in melds:
            remaining = [card for card in hand if card not in meld]
            deadwood = sum(min(DEADWOOD_THRESHOLD, card % len(RANKS) + 1) for card in remaining)
            best_deadwood = min(best_deadwood, deadwood)
        
        return best_deadwood
    
    def _format_card(self, card_idx):
        """Format a card index as a readable string."""
        suit = card_idx // len(RANKS)
        rank = card_idx % len(RANKS)
        return f"{RANKS[rank]}{SUITS[suit]}"
    
    def _format_hand(self, hand):
        """Format a hand as a readable string."""
        return " ".join(self._format_card(card) for card in sorted(hand))


class RandomPlayer:
    """A player that selects random valid actions."""
    
    def __init__(self, player_id):
        """Initialize the random player."""
        self.player_id = player_id
    
    def select_action(self, state, valid_actions):
        """Select a random valid action."""
        if not valid_actions:
            return None
        return random.choice(valid_actions)


class RuleBasedPlayer:
    """A player that follows simple heuristic rules."""
    
    def __init__(self, player_id):
        """Initialize the rule-based player."""
        self.player_id = player_id
    
    def select_action(self, state, valid_actions):
        """Select action based on heuristic rules."""
        if not valid_actions:
            return None
        
        # If we can gin, do it
        if GIN in valid_actions:
            return GIN
        
        # If we can knock with low deadwood, do it
        if KNOCK in valid_actions:
            hand = state['playerHand']
            drawn_card = state['drawnCard']
            all_cards = hand + [drawn_card]
            deadwood = self._calculate_deadwood(all_cards)
            if deadwood <= 5:  # Knock if deadwood is very low
                return KNOCK
        
        # Draw phase
        if DRAW_STOCK in valid_actions or DRAW_DISCARD in valid_actions:
            # If discard pile is not empty and would form a meld or reduce deadwood, draw from discard
            if DRAW_DISCARD in valid_actions and state['discardPile']:
                top_discard = state['discardPile'][-1]
                hand = state['playerHand']
                current_deadwood = self._calculate_deadwood(hand)
                new_deadwood = self._calculate_deadwood(hand + [top_discard])
                
                # See if adding the discard forms a new meld
                current_melds = self._find_melds(hand)
                new_melds = self._find_melds(hand + [top_discard])
                
                if len(new_melds) > len(current_melds) or new_deadwood < current_deadwood:
                    return DRAW_DISCARD
            
            # Otherwise draw from stock
            if DRAW_STOCK in valid_actions:
                return DRAW_STOCK
            else:
                return DRAW_DISCARD  # Fallback if only discard is available
        
        # Discard phase - choose the highest deadwood card that doesn't break melds
        discard_actions = [a for a in valid_actions if a >= DISCARD]
        if discard_actions:
            hand = state['playerHand']
            drawn_card = state['drawnCard']
            all_cards = hand + [drawn_card]
            
            # Find melds in the current hand
            melds = self._find_melds(all_cards)
            
            best_discard = None
            best_score = -float('inf')
            
            for action in discard_actions:
                card = action - DISCARD
                
                # Skip cards that are part of melds
                in_meld = False
                for meld in melds:
                    if card in meld:
                        in_meld = True
                        break
                
                if in_meld:
                    continue
                
                # Prefer discarding high-value cards
                card_value = min(10, card % len(RANKS) + 1)
                
                # Adjust score to prefer high-value cards
                score = card_value
                
                # Check if this is the best discard so far
                if score > best_score:
                    best_score = score
                    best_discard = action
            
            # If we found a good discard, use it
            if best_discard is not None:
                return best_discard
            
            # Otherwise, just discard the drawn card or a random card
            if drawn_card is not None:
                return DISCARD + drawn_card
            else:
                return random.choice(discard_actions)
        
        # Fallback to random action
        return random.choice(valid_actions)
    
    def _find_melds(self, hand):
        """Find all possible melds in a hand."""
        # Group cards by suit and rank
        suits = [[] for _ in range(len(SUITS))]
        ranks = [[] for _ in range(len(RANKS))]
        
        for card in hand:
            suit = card // len(RANKS)
            rank = card % len(RANKS)
            suits[suit].append(card)
            ranks[rank].append(card)
        
        melds = []
        
        # Find sets (same rank, different suits)
        for rank in range(len(RANKS)):
            if len(ranks[rank]) >= 3:
                melds.append(sorted(ranks[rank]))
        
        # Find runs (same suit, consecutive ranks)
        for suit in range(len(SUITS)):
            suit_cards = sorted([card % len(RANKS) for card in suits[suit]])
            run = []
            
            for i, rank in enumerate(suit_cards):
                if i > 0 and rank != suit_cards[i-1] + 1:
                    if len(run) >= 3:
                        melds.append([suit * len(RANKS) + r for r in run])
                    run = []
                run.append(rank)
            
            if len(run) >= 3:
                melds.append([suit * len(RANKS) + r for r in run])
        
        return melds
    
    def _calculate_deadwood(self, hand):
        """Calculate the deadwood value of a hand."""
        melds = self._find_melds(hand)
        
        # If no melds, all cards are deadwood
        if not melds:
            return sum(min(DEADWOOD_THRESHOLD, card % len(RANKS) + 1) for card in hand)
        
        # Try different combinations of non-overlapping melds
        best_deadwood = float('inf')
        
        for meld in melds:
            remaining = [card for card in hand if card not in meld]
            deadwood = sum(min(DEADWOOD_THRESHOLD, card % len(RANKS) + 1) for card in remaining)
            best_deadwood = min(best_deadwood, deadwood)
        
        return best_deadwood


class ModelPlayer:
    """A player that uses a trained model to select actions."""
    
    def __init__(self, player_id, model_type, model_path, device='cpu'):
        """Initialize the model-based player."""
        self.player_id = player_id
        self.model_type = model_type
        self.device = device
        
        # Load the appropriate model
        if model_type == 'enhanced_reinforce':
            self.agent = EnhancedREINFORCEAgent()
            checkpoint = torch.load(model_path, map_location=device)
            self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.policy.eval()
        elif model_type == 'enhanced_dqn':
            self.agent = EnhancedDQNAgent()
            checkpoint = torch.load(model_path, map_location=device)
            self.agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.policy_net.eval()
        elif model_type == 'fast_reinforce':
            self.agent = FastREINFORCEAgent()
            checkpoint = torch.load(model_path, map_location=device)
            self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.policy.eval()
        elif model_type == 'fast_dqn':
            self.agent = FastDQNAgent()
            checkpoint = torch.load(model_path, map_location=device)
            self.agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.policy_net.eval()
        elif model_type == 'mcts':
            if 'policy' in model_path:
                self.agent = MCTSAgent(policy_net=MCTSPolicyNetwork().to(device))
                checkpoint = torch.load(model_path, map_location=device)
                self.agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
                self.agent.policy_net.eval()
            elif 'value' in model_path:
                self.agent = MCTSAgent(value_net=MCTSValueNetwork().to(device))
                checkpoint = torch.load(model_path, map_location=device)
                self.agent.value_net.load_state_dict(checkpoint['value_state_dict'])
                self.agent.value_net.eval()
        elif model_type == 'reinforce':
            self.agent = REINFORCEAgent()
            checkpoint = torch.load(model_path, map_location=device)
            self.agent.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.policy.eval()
        elif model_type == 'dqn':
            self.agent = DQNAgent()
            checkpoint = torch.load(model_path, map_location=device)
            self.agent.policy_net.load_state_dict(checkpoint['policy_state_dict'])
            self.agent.policy_net.eval()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def create_hand_matrix(self, cards):
        """Create a 4x13 matrix representation of a hand."""
        matrix = np.zeros((4, 13), dtype=np.float32)
        for card in cards:
            suit = card // 13
            rank = card % 13
            matrix[suit, rank] = 1.0
        return matrix
    
    def create_discard_history(self, discards):
        """Create a 4x13 matrix for discard history."""
        matrix = np.zeros((4, 13), dtype=np.float32)
        for card in discards:
            suit = card // 13
            rank = card % 13
            matrix[suit, rank] = 1.0
        return matrix
    
    def select_action(self, state, valid_actions):
        """Select an action using the loaded model."""
        if not valid_actions:
            return None
            
        with torch.no_grad():
            # Create input tensors
            hand_matrix = torch.FloatTensor(self.create_hand_matrix(state['playerHand']))  # Shape: [4, 13]
            discard_history = torch.FloatTensor(self.create_discard_history(state['discardPile']))  # Shape: [4, 13]
            
            # Create valid actions mask
            valid_actions_mask = torch.zeros(110, dtype=torch.bool)
            for action in valid_actions:
                valid_actions_mask[action] = True
            
            # Move tensors to device
            hand_matrix = hand_matrix.to(self.device)
            discard_history = discard_history.to(self.device)
            valid_actions_mask = valid_actions_mask.to(self.device)
            
            # Add batch dimension for model input
            hand_matrix = hand_matrix.unsqueeze(0)  # Shape: [1, 4, 13]
            discard_history = discard_history.unsqueeze(0)  # Shape: [1, 4, 13]
            valid_actions_mask = valid_actions_mask.unsqueeze(0)  # Shape: [1, 110]
            
            if self.model_type in ['enhanced_reinforce', 'fast_reinforce', 'reinforce']:
                # Create opponent model tensor (zeros for now)
                opponent_model = torch.zeros(52, device=self.device)
                opponent_model = opponent_model.unsqueeze(0)  # Shape: [1, 52]
                
                # Get action probabilities
                if self.model_type == 'enhanced_reinforce':
                    probs = self.agent.policy(
                        hand_matrix,
                        discard_history,
                        opponent_model,
                        valid_actions_mask
                    )
                else:  # fast_reinforce or reinforce
                    probs = self.agent.policy(
                        hand_matrix,
                        discard_history,
                        valid_actions_mask
                    )
                
                # Select action with highest probability among valid actions
                action_probs = probs.squeeze(0)
                action_probs[~valid_actions_mask[0]] = float('-inf')
                action = action_probs.argmax().item()
            
            elif self.model_type in ['enhanced_dqn', 'fast_dqn', 'dqn']:
                # Get Q-values
                if self.model_type == 'enhanced_dqn':
                    q_values = self.agent.policy_net(
                        hand_matrix,
                        discard_history,
                        valid_actions_mask
                    )
                else:  # fast_dqn or dqn
                    q_values = self.agent.policy_net(
                        hand_matrix,
                        discard_history,
                        valid_actions_mask
                    )
                
                # Select action with highest Q-value among valid actions
                q_values = q_values.squeeze(0)
                q_values[~valid_actions_mask[0]] = float('-inf')
                action = q_values.argmax().item()
            
            elif self.model_type == 'mcts':
                # For MCTS, we need both policy and value predictions
                if hasattr(self.agent, 'policy_net'):
                    policy_output = self.agent.policy_net(
                        hand_matrix,
                        discard_history,
                        valid_actions_mask
                    )
                    action_probs = policy_output.squeeze(0)
                    action_probs[~valid_actions_mask[0]] = float('-inf')
                    action = action_probs.argmax().item()
                else:  # Using value network
                    # For value network, we'll evaluate each action and choose the best
                    best_value = float('-inf')
                    action = valid_actions[0]  # Default to first valid action
                    
                    for valid_action in valid_actions:
                        # Create a hypothetical next state
                        next_hand_matrix = hand_matrix.clone()
                        next_discard_history = discard_history.clone()
                        
                        # Update matrices based on action
                        if valid_action >= DISCARD:  # Discard action
                            card = valid_action - DISCARD
                            suit = card // 13
                            rank = card % 13
                            next_hand_matrix[0, suit, rank] = 0
                            next_discard_history[0, suit, rank] = 1
                        
                        # Get value prediction
                        value = self.agent.value_net(
                            next_hand_matrix,
                            next_discard_history,
                            valid_actions_mask
                        ).item()
                        
                        if value > best_value:
                            best_value = value
                            action = valid_action
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
        
        return action
    
    def _format_hand(self, cards):
        """Format a list of cards for display."""
        if not cards:
            return "Empty"
        card_strs = []
        for card in sorted(cards):
            suit = card // 13
            rank = card % 13
            card_strs.append(f"{RANKS[rank]}{SUITS[suit]}")
        return " ".join(card_strs)
    
    def _calculate_deadwood(self, hand):
        """Calculate the deadwood value of a hand."""
        # Find all possible melds in the hand
        melds = self._find_melds(hand)
        
        # If no melds, all cards are deadwood
        if not melds:
            return sum(min(10, card % 13 + 1) for card in hand)
        
        # Try different combinations of non-overlapping melds
        best_deadwood = float('inf')
        
        for meld in melds:
            # Calculate deadwood for cards not in this meld
            remaining = [card for card in hand if card not in meld]
            deadwood = sum(min(10, card % 13 + 1) for card in remaining)
            best_deadwood = min(best_deadwood, deadwood)
        
        return best_deadwood
    
    def _find_melds(self, hand):
        """Find all possible melds in a hand."""
        if not hand:
            return []
            
        # Group cards by suit and rank
        suits = [[] for _ in range(4)]
        ranks = [[] for _ in range(13)]
        
        for card in hand:
            suit = card // 13
            rank = card % 13
            suits[suit].append(card)
            ranks[rank].append(card)
        
        melds = []
        
        # Find sets (same rank, different suits)
        for rank_cards in ranks:
            if len(rank_cards) >= 3:
                melds.append(sorted(rank_cards))
        
        # Find runs (consecutive ranks in same suit)
        for suit_cards in suits:
            if len(suit_cards) < 3:
                continue
                
            # Sort by rank
            suit_cards.sort(key=lambda x: x % 13)
            
            # Find runs
            current_run = [suit_cards[0]]
            for i in range(1, len(suit_cards)):
                current_rank = suit_cards[i] % 13
                prev_rank = suit_cards[i-1] % 13
                
                if current_rank == prev_rank + 1:
                    # Consecutive rank
                    current_run.append(suit_cards[i])
                else:
                    # End of run
                    if len(current_run) >= 3:
                        melds.append(current_run.copy())
                    current_run = [suit_cards[i]]
            
            # Check last run
            if len(current_run) >= 3:
                melds.append(current_run)
        
        return melds


def run_gameplay_evaluation(models_to_evaluate, opponents, num_games=NUM_GAMES, verbose=False):
    """Run games between the trained models and opponents."""
    results = {}
    
    # Enable verbose mode for debugging
    VERBOSE = True  # Force verbose mode on to see more output
    
    # Setup device for model inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # For Apple Silicon
    print(f"Using device: {device}")
    
    # Reduce the number of games for debugging
    if num_games > 10:
        print(f"Reducing number of games from {num_games} to 5 for debugging")
        num_games = 5
        
    # Create a game environment with verbose output
    game = GinRummyGame(verbose=VERBOSE)
    
    # For each model, play against each opponent
    for model_info in tqdm(models_to_evaluate, desc="Evaluating models"):
        model_name = model_info['name']
        model_type = model_info['type']
        model_path = model_info['path']
        
        print(f"\nEvaluating {model_name} ({model_type}) from {model_path}")
        
        # Initialize model results
        model_results = {}
        
        try:
            # Create model player
            model_player = ModelPlayer(0, model_type, model_path, device)
            
            # Play against each opponent
            for opponent_info in opponents:
                opponent_name = opponent_info['name']
                opponent_type = opponent_info['type']
                
                print(f"  Playing against {opponent_name}")
                
                # Initialize metrics
                wins = 0
                losses = 0
                draws = 0
                total_score = 0
                total_turns = 0
                games_played = 0
                
                # Play the specified number of games
                for game_idx in tqdm(range(num_games), desc=f"Games vs {opponent_name}", leave=False):
                    try:
                        # Reset the game
                        game.reset()
                        
                        # Create opponent (player 1)
                        if opponent_type == 'random':
                            opponent = RandomPlayer(1)
                        elif opponent_type == 'rule_based':
                            opponent = RuleBasedPlayer(1)
                        else:
                            # Another model as opponent
                            opponent = ModelPlayer(1, opponent_info['model_type'], opponent_info['model_path'], device)
                        
                        # Play until game over
                        while not game.game_over:
                            # Get current player
                            current_player_id = game.current_player
                            
                            # Get current state and valid actions
                            state = game.get_state(current_player_id)
                            valid_actions = state['validActions']
                            
                            # Select action based on current player
                            if current_player_id == 0:  # Model player
                                action = model_player.select_action(state, valid_actions)
                            else:  # Opponent
                                action = opponent.select_action(state, valid_actions)
                            
                            # If no valid action, end the game in a draw
                            if action is None:
                                game.game_over = True
                                game.winner = None
                                break
                            
                            # Apply action to the game
                            state, reward, done = game.step(current_player_id, action)
                        
                        # Record game result
                        games_played += 1
                        total_turns += game.turn_count
                        
                        if game.winner is None:
                            draws += 1
                        elif game.winner == 0:
                            wins += 1
                            # Calculate approximate score
                            if game.drawn_card is None:
                                # Game ended with a knock or gin
                                opponent_deadwood = game.calculate_deadwood(game.player_hands[1])
                                player_deadwood = game.calculate_deadwood(game.player_hands[0])
                                total_score += (opponent_deadwood - player_deadwood)
                            else:
                                # Game ended without a proper finish
                                total_score += 1
                        else:
                            losses += 1
                            # Calculate approximate score
                            if game.drawn_card is None:
                                # Game ended with a knock or gin
                                opponent_deadwood = game.calculate_deadwood(game.player_hands[1])
                                player_deadwood = game.calculate_deadwood(game.player_hands[0])
                                total_score -= (player_deadwood - opponent_deadwood)
                            else:
                                # Game ended without a proper finish
                                total_score -= 1
                    
                    except Exception as e:
                        print(f"Error in game {game_idx}: {e}")
                        # Continue with next game
                        continue
                
                # Calculate metrics
                win_rate = wins / games_played if games_played > 0 else 0
                avg_score = total_score / games_played if games_played > 0 else 0
                avg_turns = total_turns / games_played if games_played > 0 else 0
                
                # Store results
                model_results[opponent_name] = {
                    'wins': wins,
                    'losses': losses,
                    'draws': draws,
                    'games_played': games_played,
                    'win_rate': win_rate,
                    'avg_score': avg_score,
                    'avg_turns': avg_turns
                }
                
                print(f"  Results vs {opponent_name}: {wins} wins, {losses} losses, {draws} draws")
                print(f"  Win rate: {win_rate:.2f}, Avg score: {avg_score:.2f}, Avg turns: {avg_turns:.2f}")
        
        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            continue
        
        # Store results for this model
        results[model_name] = model_results
    
    return results


def visualize_results(results):
    """Visualize the evaluation results using matplotlib."""
    if not results:
        print("No results to visualize")
        return
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Collect data for plotting
    model_names = list(results.keys())
    opponent_names = set()
    for model_results in results.values():
        opponent_names.update(model_results.keys())
    opponent_names = sorted(list(opponent_names))
    
    # Plot win rates
    win_rates = {}
    for opponent in opponent_names:
        win_rates[opponent] = []
        for model in model_names:
            if opponent in results[model]:
                win_rates[opponent].append(results[model][opponent]['win_rate'])
            else:
                win_rates[opponent].append(0)
    
    # Create bar chart
    plt.subplot(2, 1, 1)
    x = np.arange(len(model_names))
    width = 0.8 / len(opponent_names)
    
    for i, opponent in enumerate(opponent_names):
        plt.bar(x + i * width - width * len(opponent_names) / 2, win_rates[opponent], 
                width=width, label=f'vs {opponent}')
    
    plt.xlabel('Model')
    plt.ylabel('Win Rate')
    plt.title('Win Rates Against Different Opponents')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Plot average scores
    avg_scores = {}
    for opponent in opponent_names:
        avg_scores[opponent] = []
        for model in model_names:
            if opponent in results[model]:
                avg_scores[opponent].append(results[model][opponent]['avg_score'])
            else:
                avg_scores[opponent].append(0)
    
    plt.subplot(2, 1, 2)
    
    for i, opponent in enumerate(opponent_names):
        plt.bar(x + i * width - width * len(opponent_names) / 2, avg_scores[opponent], 
                width=width, label=f'vs {opponent}')
    
    plt.xlabel('Model')
    plt.ylabel('Average Score')
    plt.title('Average Scores Against Different Opponents')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_results.png')
    print("Results visualization saved to model_evaluation_results.png")
    
    # Also show the plot if not running in headless environment
    try:
        plt.show()
    except:
        pass


def main():
    """Main function to run the evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Gin Rummy models')
    parser.add_argument('--games', type=int, default=NUM_GAMES, help='Number of games to play for each evaluation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Setup models to evaluate - automatically find model files
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"Error: Models directory '{models_dir}' not found")
        return
    
    # Look for model files
    model_files = {
        'dqn': [],
        'reinforce': [],
        'mcts': []
    }
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.pt'):
            filepath = os.path.join(models_dir, filename)
            if 'dqn' in filename.lower():
                model_files['dqn'].append(filepath)
            elif 'reinforce' in filename.lower():
                model_files['reinforce'].append(filepath)
            elif 'mcts' in filename.lower():
                model_files['mcts'].append(filepath)
    
    # Create model configurations
    models_to_evaluate = []
    
    # DQN models
    for model_path in model_files['dqn']:
        model_name = os.path.basename(model_path).replace('.pt', '')
        if 'quick' in model_path.lower():
            models_to_evaluate.append({
                'name': model_name,
                'type': 'fast_dqn',
                'path': model_path
            })
        elif 'enhanced' in model_path.lower():
            models_to_evaluate.append({
                'name': model_name,
                'type': 'enhanced_dqn',
                'path': model_path
            })
        else:
            models_to_evaluate.append({
                'name': model_name,
                'type': 'dqn',
                'path': model_path
            })
    
    # REINFORCE models
    for model_path in model_files['reinforce']:
        model_name = os.path.basename(model_path).replace('.pt', '')
        if 'quick' in model_path.lower():
            models_to_evaluate.append({
                'name': model_name,
                'type': 'fast_reinforce',
                'path': model_path
            })
        elif 'enhanced' in model_path.lower():
            models_to_evaluate.append({
                'name': model_name,
                'type': 'enhanced_reinforce',
                'path': model_path
            })
        else:
            models_to_evaluate.append({
                'name': model_name,
                'type': 'reinforce',
                'path': model_path
            })
    
    # MCTS models
    for model_path in model_files['mcts']:
        model_name = os.path.basename(model_path).replace('.pt', '')
        models_to_evaluate.append({
            'name': model_name,
            'type': 'mcts',
            'path': model_path
        })
    
    if not models_to_evaluate:
        print("No model files found to evaluate")
        return
    
    print(f"Found {len(models_to_evaluate)} models to evaluate:")
    for model in models_to_evaluate:
        print(f"  - {model['name']} ({model['type']}): {model['path']}")
    
    # Setup opponents
    opponents = [
        {'name': 'Random', 'type': 'random'},
        {'name': 'Rule-Based', 'type': 'rule_based'}
    ]
    
    # Run evaluation
    results = run_gameplay_evaluation(models_to_evaluate, opponents, args.games, args.verbose)
    
    # Visualize results
    visualize_results(results)
    
    # Save results to a file
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to evaluation_results.json")


if __name__ == "__main__":
    main() 