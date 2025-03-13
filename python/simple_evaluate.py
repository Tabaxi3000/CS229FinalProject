import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import copy
import json

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
        
        # If no card has been drawn, can only draw from stock or discard
        if self.drawn_card is None:
            if self.deck:
                valid_actions.append(DRAW_STOCK)
            if self.discard_pile:
                valid_actions.append(DRAW_DISCARD)
        else:
            # If a card has been drawn, can discard any card in hand or the drawn card
            # Add discard actions for cards in the player's hand
            for card in self.player_hands[player_id]:
                # Ensure card is an integer and calculate correct discard action
                card_idx = int(card)
                discard_action = DISCARD + card_idx
                if self.verbose:
                    print(f"Adding discard action {discard_action} for card {self._format_card(card_idx)} from hand")
                valid_actions.append(discard_action)
            
            # Add the drawn card as a valid discard
            drawn_card_idx = int(self.drawn_card)
            drawn_discard_action = DISCARD + drawn_card_idx
            if self.verbose:
                print(f"Adding discard action {drawn_discard_action} for drawn card {self._format_card(drawn_card_idx)}")
            valid_actions.append(drawn_discard_action)
            
            # Check if player can knock or gin
            temp_hand = self.player_hands[player_id] + [self.drawn_card]
            deadwood = self.calculate_deadwood(temp_hand)
            
            if deadwood <= DEADWOOD_THRESHOLD:
                valid_actions.append(KNOCK)
                if self.verbose:
                    print(f"Adding KNOCK action (deadwood: {deadwood})")
                
            if deadwood == 0:
                valid_actions.append(GIN)
                if self.verbose:
                    print(f"Adding GIN action (deadwood: {deadwood})")
        
        if self.verbose:
            print(f"Valid actions for player {player_id+1}: {valid_actions}")
            if self.drawn_card is not None:
                print(f"Current hand: {[self._format_card(c) for c in self.player_hands[player_id]]}")
                print(f"Drawn card: {self._format_card(self.drawn_card)}")
        
        return valid_actions
    
    def step(self, player_id, action):
        # Validate player's turn
        if player_id != self.current_player:
            if self.verbose:
                print(f"Not player {player_id}'s turn!")
            return self.get_state(player_id), -1, True
        
        # Check if game is over
        if self.game_over:
            if self.verbose:
                print("Game is already over")
            return self.get_state(player_id), 0, True
            
        # Get valid actions
        valid_actions = self.get_valid_actions(player_id)
        if action not in valid_actions:
            if self.verbose:
                print(f"Invalid action {action}!")
            return self.get_state(player_id), -1, True
            
        # Convert action to int to ensure type consistency
        action = int(action)
        
        # Handle the action
        if action == DRAW_STOCK:
            if len(self.deck) == 0:
                # No cards left in stock pile
                return self.get_state(player_id), 0, True
            self.drawn_card = self.deck.pop()
            if self.verbose:
                print(f"Player {player_id+1} draws {self._format_card(self.drawn_card)} from stock")
                
        elif action == DRAW_DISCARD:
            if len(self.discard_pile) == 0:
                return self.get_state(player_id), -1, True
            self.drawn_card = self.discard_pile.pop()
            if self.verbose:
                print(f"Player {player_id+1} draws {self._format_card(self.drawn_card)} from discard")
                
        elif DISCARD <= action < KNOCK:
            # Handle discard action
            card_to_discard = action - DISCARD
            
            # Check if card is in hand or is the drawn card
            if card_to_discard == self.drawn_card:
                self.discard_pile.append(card_to_discard)
                self.drawn_card = None
            elif card_to_discard in self.player_hands[player_id]:
                self.player_hands[player_id].remove(card_to_discard)
                self.discard_pile.append(card_to_discard)
            else:
                return self.get_state(player_id), -1, True
                
            if self.verbose:
                print(f"Player {player_id+1} discards {self._format_card(card_to_discard)}")
                
            # After a successful discard, move to next player's turn
            self.current_player = 1 - player_id
            self.drawn_card = None
            self.turn_count += 1
            
        elif action == KNOCK or action == GIN:
            # Handle knock/gin action
            deadwood = self.calculate_deadwood(self.player_hands[player_id])
            if action == GIN and deadwood == 0:
                self.winner = player_id
                return self.get_state(player_id), 1, True
            elif action == KNOCK and deadwood <= DEADWOOD_THRESHOLD:
                self.winner = player_id
                return self.get_state(player_id), 1, True
            else:
                return self.get_state(player_id), -1, True
                
        # Check turn limit
        if self.turn_count >= MAX_TURNS:
            self.game_over = True
            self.winner = None  # Draw
            if self.verbose:
                print("Game ends in a draw - turn limit reached")
        
        return self.get_state(player_id), 0, self.game_over
    
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


class GreedyPlayer:
    """A player that tries to minimize deadwood aggressively."""
    
    def __init__(self, player_id):
        """Initialize the greedy player."""
        self.player_id = player_id
    
    def select_action(self, state, valid_actions):
        """Select action based on greedy strategy."""
        if not valid_actions:
            return None
        
        # If we can gin, do it
        if GIN in valid_actions:
            return GIN
        
        # If we can knock, always do it
        if KNOCK in valid_actions:
            return KNOCK
        
        # Draw phase
        if DRAW_STOCK in valid_actions or DRAW_DISCARD in valid_actions:
            # Always prefer drawing from discard pile if it reduces deadwood
            if DRAW_DISCARD in valid_actions and state['discardPile']:
                top_discard = state['discardPile'][-1]
                hand = state['playerHand']
                current_deadwood = self._calculate_deadwood(hand)
                new_deadwood = self._calculate_deadwood(hand + [top_discard])
                
                if new_deadwood <= current_deadwood:
                    return DRAW_DISCARD
            
            # Otherwise draw from stock
            if DRAW_STOCK in valid_actions:
                return DRAW_STOCK
            else:
                return DRAW_DISCARD
        
        # Discard phase - always choose the highest deadwood card
        discard_actions = [a for a in valid_actions if a >= DISCARD]
        if discard_actions:
            hand = state['playerHand']
            drawn_card = state['drawnCard']
            
            best_discard = None
            min_deadwood = float('inf')
            
            for action in discard_actions:
                card = action - DISCARD
                
                # Check how removing this card affects deadwood
                if card == drawn_card:
                    # If discarding drawn card, just keep current hand
                    test_hand = hand.copy()
                else:
                    # If discarding from hand, replace with drawn card
                    test_hand = [c for c in hand if c != card] + [drawn_card]
                
                deadwood = self._calculate_deadwood(test_hand)
                
                # Prefer discarding higher cards if deadwood is the same
                card_value = min(10, card % len(RANKS) + 1)
                adjusted_deadwood = deadwood - (card_value / 100.0)
                
                if adjusted_deadwood < min_deadwood:
                    min_deadwood = adjusted_deadwood
                    best_discard = action
            
            return best_discard
        
        # Fallback to random action
        return random.choice(valid_actions)
    
    def _calculate_deadwood(self, hand):
        """Calculate the deadwood value of a hand."""
        if not hand:
            return 0
            
        melds = self._find_melds(hand)
        
        # If no melds, all cards are deadwood
        if not melds:
            return sum(min(10, card % len(RANKS) + 1) for card in hand)
        
        # Try different combinations of non-overlapping melds
        best_deadwood = float('inf')
        
        for meld in melds:
            remaining = [card for card in hand if card not in meld]
            deadwood = sum(min(10, card % len(RANKS) + 1) for card in remaining)
            best_deadwood = min(best_deadwood, deadwood)
        
        return best_deadwood
    
    def _find_melds(self, hand):
        """Find all possible melds in a hand."""
        # Similar implementation as RuleBasedPlayer
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


def run_gameplay_evaluation(players_to_evaluate, num_games=NUM_GAMES, verbose=False):
    """Run games between different players."""
    results = {}
    
    # Create a game environment
    game = GinRummyGame(verbose=verbose)
    
    # For each player, play against each other player
    for i, player1_info in enumerate(players_to_evaluate):
        player1_name = player1_info['name']
        player1_type = player1_info['type']
        
        print(f"\nEvaluating {player1_name}")
        
        # Initialize player results
        player_results = {}
        
        # Play against each opponent
        for j, player2_info in enumerate(players_to_evaluate):
            if i == j:  # Skip playing against itself
                continue
                
            player2_name = player2_info['name']
            player2_type = player2_info['type']
            
            print(f"  Playing against {player2_name}")
            
            # Initialize metrics
            wins = 0
            losses = 0
            draws = 0
            total_score = 0
            total_turns = 0
            games_played = 0
            
            # Play the specified number of games
            for game_idx in tqdm(range(num_games), desc=f"Games vs {player2_name}", leave=False):
                try:
                    # Reset the game
                    game.reset()
                    
                    # Create players
                    if player1_type == 'random':
                        player1 = RandomPlayer(0)
                    elif player1_type == 'rule_based':
                        player1 = RuleBasedPlayer(0)
                    elif player1_type == 'greedy':
                        player1 = GreedyPlayer(0)
                    else:
                        raise ValueError(f"Unknown player type: {player1_type}")
                    
                    if player2_type == 'random':
                        player2 = RandomPlayer(1)
                    elif player2_type == 'rule_based':
                        player2 = RuleBasedPlayer(1)
                    elif player2_type == 'greedy':
                        player2 = GreedyPlayer(1)
                    else:
                        raise ValueError(f"Unknown player type: {player2_type}")
                    
                    # Play until game over
                    while not game.game_over:
                        # Get current player
                        current_player_id = game.current_player
                        
                        # Get current state and valid actions
                        state = game.get_state(current_player_id)
                        valid_actions = state['validActions']
                        
                        # Select action based on current player
                        if current_player_id == 0:  # Player 1
                            action = player1.select_action(state, valid_actions)
                        else:  # Player 2
                            action = player2.select_action(state, valid_actions)
                        
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
            player_results[player2_name] = {
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'games_played': games_played,
                'win_rate': win_rate,
                'avg_score': avg_score,
                'avg_turns': avg_turns
            }
            
            print(f"  Results vs {player2_name}: {wins} wins, {losses} losses, {draws} draws")
            print(f"  Win rate: {win_rate:.2f}, Avg score: {avg_score:.2f}, Avg turns: {avg_turns:.2f}")
        
        # Store results for this player
        results[player1_name] = player_results
    
    return results


def visualize_results(results):
    """Visualize the evaluation results using matplotlib."""
    if not results:
        print("No results to visualize")
        return
    
    # Set up the plot
    plt.figure(figsize=(14, 8))
    
    # Collect data for plotting
    player_names = list(results.keys())
    opponent_names = []
    for player in player_names:
        opponent_names.extend(list(results[player].keys()))
    opponent_names = sorted(list(set(opponent_names)))
    
    # Create a heatmap for win rates
    win_rates = np.zeros((len(player_names), len(opponent_names)))
    
    for i, player in enumerate(player_names):
        for j, opponent in enumerate(opponent_names):
            if opponent in results[player]:
                win_rates[i, j] = results[player][opponent]['win_rate']
    
    plt.subplot(2, 1, 1)
    plt.imshow(win_rates, cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(label='Win Rate')
    plt.title('Win Rates Matrix')
    plt.xticks(range(len(opponent_names)), opponent_names, rotation=45, ha='right')
    plt.yticks(range(len(player_names)), player_names)
    
    # Add text annotations
    for i in range(len(player_names)):
        for j in range(len(opponent_names)):
            if opponent_names[j] in results[player_names[i]]:
                plt.text(j, i, f"{win_rates[i, j]:.2f}", ha="center", va="center", color="black")
    
    # Create a bar chart of overall performance
    plt.subplot(2, 1, 2)
    
    # Calculate average win rates for each player
    avg_win_rates = []
    for player in player_names:
        rates = [results[player][opponent]['win_rate'] for opponent in results[player]]
        avg_win_rates.append(sum(rates) / len(rates) if rates else 0)
    
    plt.bar(player_names, avg_win_rates, color='skyblue')
    plt.ylim(0, 1)
    plt.title('Average Win Rates')
    plt.ylabel('Win Rate')
    plt.xticks(rotation=45, ha='right')
    
    # Add text annotations
    for i, v in enumerate(avg_win_rates):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('player_evaluation_results.png')
    print("Results visualization saved to player_evaluation_results.png")
    
    # Also show the plot if not running in headless environment
    try:
        plt.show()
    except:
        pass


def main():
    """Main function to run the evaluation."""
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate Gin Rummy players')
    parser.add_argument('--games', type=int, default=NUM_GAMES, help='Number of games to play for each evaluation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Setup players to evaluate
    players_to_evaluate = [
        {'name': 'Random', 'type': 'random'},
        {'name': 'Rule-Based', 'type': 'rule_based'},
        {'name': 'Greedy', 'type': 'greedy'}
    ]
    
    # Run evaluation
    results = run_gameplay_evaluation(players_to_evaluate, args.games, args.verbose)
    
    # Visualize results
    visualize_results(results)
    
    # Save results to a file
    import json
    with open('simple_evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    print("Results saved to simple_evaluation_results.json")


if __name__ == "__main__":
    main() 