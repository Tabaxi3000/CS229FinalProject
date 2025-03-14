import random
import torch
import numpy as np

# Action constants
DRAW_STOCK = 0
DRAW_DISCARD = 1
DISCARD_START = 2
DISCARD_END = 53
KNOCK = 108
GIN = 109

class GinRummyEnv:
    """A simplified Gin Rummy environment for testing."""
    
    def __init__(self):
        self.current_player = 0
        self.deck = list(range(52))
        self.discard_pile = []
        self.player_hands = [[], []]
        self.phase = 'draw'  # 'draw' or 'discard'
        
    def reset(self):
        """Reset the environment to start a new game."""
        # Shuffle deck
        self.deck = list(range(52))
        random.shuffle(self.deck)
        
        # Deal cards
        self.player_hands = [[], []]
        for _ in range(10):
            self.player_hands[0].append(self.deck.pop())
            self.player_hands[1].append(self.deck.pop())
        
        # Initialize discard pile
        self.discard_pile = [self.deck.pop()]
        
        # Set starting player
        self.current_player = 0
        self.phase = 'draw'
        
        # Create state representation
        return self._get_state()
    
    def step(self, action):
        """Take an action in the environment."""
        reward = 0
        done = False
        info = {}
        
        # Process action
        if action == DRAW_STOCK:
            # Draw from stock
            if self.phase == 'draw':
                if self.deck:
                    card = self.deck.pop()
                    self.player_hands[self.current_player].append(card)
                    self.phase = 'discard'
                else:
                    # Deck is empty, game is a draw
                    done = True
            else:
                # Invalid action
                reward = -1
                done = True
                
        elif action == DRAW_DISCARD:
            # Draw from discard
            if self.phase == 'draw' and self.discard_pile:
                card = self.discard_pile.pop()
                self.player_hands[self.current_player].append(card)
                self.phase = 'discard'
            else:
                # Invalid action
                reward = -1
                done = True
                
        elif DISCARD_START <= action <= DISCARD_END:
            # Discard a card
            card_idx = action - DISCARD_START
            if self.phase == 'discard' and card_idx in self.player_hands[self.current_player]:
                self.player_hands[self.current_player].remove(card_idx)
                self.discard_pile.append(card_idx)
                self.phase = 'draw'
                # Switch player
                self.current_player = 1 - self.current_player
            else:
                # Invalid action
                reward = -1
                done = True
                
        elif action == KNOCK:
            # Knock
            if self.phase == 'discard':
                deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
                if deadwood <= 10:
                    # Valid knock
                    opponent_deadwood = self._calculate_deadwood(self.player_hands[1 - self.current_player])
                    if deadwood < opponent_deadwood:
                        reward = 1  # Win
                    elif deadwood > opponent_deadwood:
                        reward = -1  # Loss
                    else:
                        reward = 0  # Draw
                    done = True
                else:
                    # Invalid knock (deadwood > 10)
                    reward = -1
                    done = True
            else:
                # Invalid action
                reward = -1
                done = True
                
        elif action == GIN:
            # Gin
            if self.phase == 'discard':
                deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
                if deadwood == 0:
                    # Valid gin
                    reward = 1
                    done = True
                else:
                    # Invalid gin (deadwood > 0)
                    reward = -1
                    done = True
            else:
                # Invalid action
                reward = -1
                done = True
        
        # Return next state, reward, done, info
        return self._get_state(), reward, done, info
    
    def _get_state(self):
        """Get the current state representation."""
        # Create hand matrix (4x13 for each player)
        hand_matrix = torch.zeros(1, 4, 13)
        for card in self.player_hands[self.current_player]:
            suit = card // 13
            rank = card % 13
            hand_matrix[0, suit, rank] = 1
        
        # Create discard history
        # Create a batch dimension and ensure it's 3D: [batch_size, seq_len, features]
        discard_history = torch.zeros(1, 10, 52)  # [batch_size, seq_len, card_idx]
        for i, card in enumerate(self.discard_pile[-10:]):
            discard_history[0, i, card] = 1
        
        # Create valid actions mask
        valid_actions_mask = torch.zeros(110)
        
        if self.phase == 'draw':
            # Can draw from stock or discard
            valid_actions_mask[DRAW_STOCK] = 1
            if self.discard_pile:
                valid_actions_mask[DRAW_DISCARD] = 1
        else:
            # Can discard any card in hand
            for card in self.player_hands[self.current_player]:
                valid_actions_mask[DISCARD_START + card] = 1
            
            # Check if can knock or gin
            deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
            if deadwood <= 10:
                valid_actions_mask[KNOCK] = 1
            if deadwood == 0:
                valid_actions_mask[GIN] = 1
        
        # Create opponent model (for MCTS)
        opponent_model = torch.zeros(52)
        for card in self.player_hands[1 - self.current_player]:
            opponent_model[card] = 1
        
        return {
            'hand_matrix': hand_matrix,
            'discard_history': discard_history,
            'valid_actions_mask': valid_actions_mask,
            'current_player': self.current_player,
            'phase': self.phase,
            'opponent_model': opponent_model
        }
    
    def _calculate_deadwood(self, hand):
        """Calculate deadwood points for a hand."""
        if not hand:
            return 0
        
        # Find all possible melds
        melds = self._find_melds(hand)
        
        # Calculate deadwood from cards not in melds
        deadwood_cards = set(hand)
        for meld in melds:
            for card in meld:
                if card in deadwood_cards:
                    deadwood_cards.remove(card)
        
        # Calculate points (face cards = 10, ace = 1, others = value)
        points = 0
        for card in deadwood_cards:
            rank = card % 13
            if rank >= 10:  # Jack, Queen, King
                points += 10
            else:  # Ace-10
                points += rank + 1
        
        return points

    def _find_melds(self, hand):
        """Find all valid melds in a hand."""
        if not hand:
            return []
        
        # Find sets (same rank, different suits)
        ranks = [[] for _ in range(13)]
        for card in hand:
            rank = card % 13
            ranks[rank].append(card)
        
        # Find runs (same suit, consecutive ranks)
        suits = [[] for _ in range(4)]
        for card in hand:
            suit = card // 13
            suits[suit].append(card)
        
        sets = []
        for rank_cards in ranks:
            if len(rank_cards) >= 3:
                sets.append(rank_cards)
        
        runs = []
        for suit_cards in suits:
            if len(suit_cards) >= 3:
                # Sort by rank
                suit_cards.sort(key=lambda x: x % 13)
                # Find consecutive sequences
                run = [suit_cards[0]]
                for i in range(1, len(suit_cards)):
                    if suit_cards[i] % 13 == (suit_cards[i-1] % 13) + 1:
                        run.append(suit_cards[i])
                    else:
                        if len(run) >= 3:
                            runs.append(run.copy())
                        run = [suit_cards[i]]
                if len(run) >= 3:
                    runs.append(run)
        
        return sets + runs 

    def print_state(self):
        """Print the current game state in a human-readable format."""
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        def card_to_str(card_idx):
            suit = card_idx // 13
            rank = card_idx % 13
            return f"{ranks[rank]}{suits[suit]}"
        
        print("\n=== Current Game State ===")
        print(f"Current player: {self.current_player}")
        print(f"Phase: {self.phase}")
        
        # Print player hands
        for player in range(2):
            hand = sorted(self.player_hands[player], key=lambda x: (x // 13, x % 13))
            hand_str = ' '.join(card_to_str(card) for card in hand)
            if player == self.current_player:
                print(f"Player {player} hand (current): {hand_str}")
            else:
                print(f"Player {player} hand: {hand_str}")
        
        # Print discard pile
        if self.discard_pile:
            top_discard = self.discard_pile[-1]
            print(f"Top discard: {card_to_str(top_discard)}")
        else:
            print("Discard pile: empty")
        
        # Print remaining cards in deck
        print(f"Cards in deck: {len(self.deck)}")
        
        # Print deadwood for current player
        deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
        print(f"Current player deadwood: {deadwood}")
        
        # Print valid actions
        valid_actions = []
        if self.phase == 'draw':
            valid_actions.append(f"Draw from stock ({DRAW_STOCK})")
            if self.discard_pile:
                valid_actions.append(f"Draw from discard ({DRAW_DISCARD})")
        else:
            for card in self.player_hands[self.current_player]:
                valid_actions.append(f"Discard {card_to_str(card)} ({DISCARD_START + card})")
            
            if deadwood <= 10:
                valid_actions.append(f"Knock ({KNOCK})")
            if deadwood == 0:
                valid_actions.append(f"Gin ({GIN})")
        
        print("Valid actions:")
        for action in valid_actions:
            print(f"  {action}")
        
        print("=========================") 