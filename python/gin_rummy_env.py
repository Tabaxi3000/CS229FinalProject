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
        self.deck = list(range(52))
        self.hands = [[], []]
        self.discard = []
        self.stock = []
        self.current = 0
        self.phase = 'draw'
        self.reset()
        
    def reset(self):
        """Reset the environment to start a new game."""
        self.deck = list(range(52))
        random.shuffle(self.deck)
        
        self.hands = [self.deck[:10], self.deck[10:20]]
        self.discard = self.deck[20:21]
        self.stock = self.deck[21:]
        
        self.current = 0
        self.phase = 'draw'
        
        return self._get_state()
    
    def step(self, action):
        """Take an action in the environment."""
        if not self._is_valid(action):
            return self._get_state(), -50, True
            
        reward = 0
        done = False
        
        if action == 0:  # Draw from stock
            if not self.stock:
                return self._get_state(), 0, True
            card = self.stock.pop(0)
            self.hands[self.current].append(card)
            self.phase = 'discard'
            
        elif action == 1:  # Draw from discard
            if not self.discard:
                return self._get_state(), 0, True
            card = self.discard.pop()
            self.hands[self.current].append(card)
            self.phase = 'discard'
            
        elif 2 <= action <= 53:  # Discard
            card = action - 2
            if card not in self.hands[self.current]:
                return self._get_state(), -50, True
            self.hands[self.current].remove(card)
            self.discard.append(card)
            self.phase = 'draw'
            self.current = 1 - self.current
            
        elif action == 108:  # Knock
            deadwood = self._get_deadwood(self.hands[self.current])
            if deadwood <= 10:
                opp_deadwood = self._get_deadwood(self.hands[1 - self.current])
                if deadwood < opp_deadwood:
                    reward = 10
                else:
                    reward = -10
            else:
                reward = -25
            done = True
            
        elif action == 109:  # Gin
            deadwood = self._get_deadwood(self.hands[self.current])
            if deadwood == 0:
                reward = 25
            else:
                reward = -25
            done = True
            
        return self._get_state(), reward, done
    
    def _get_state(self):
        """Get the current state representation."""
        hand = np.zeros((4, 13))
        for card in self.hands[self.current]:
            suit, rank = divmod(card, 13)
            hand[suit, rank] = 1
            
        discard_top = self.discard[-1] if self.discard else -1
        
        valid = np.zeros(110, dtype=bool)
        if self.phase == 'draw':
            if self.stock:
                valid[0] = True
            if self.discard:
                valid[1] = True
        else:
            deadwood = self._get_deadwood(self.hands[self.current])
            for card in self.hands[self.current]:
                valid[card + 2] = True
            if deadwood == 0:
                valid[109] = True
            elif deadwood <= 10:
                valid[108] = True
                
        return {
            'hand_matrix': hand[None, ...],
            'discard_top': discard_top,
            'valid_actions_mask': valid,
            'phase': self.phase,
            'current_player': self.current
        }
    
    def _get_deadwood(self, hand):
        """Calculate deadwood points for a hand."""
        if not hand:
            return float('inf')
        
        melds = []
        ranks = {}
        suits = {}
        
        for card in hand:
            rank = card % 13
            suit = card // 13
            ranks.setdefault(rank, []).append(card)
            suits.setdefault(suit, []).append(card)
        
        for cards in ranks.values():
            if len(cards) >= 3:
                melds.append(cards[:])
        
        for cards in suits.values():
            cards.sort(key=lambda x: x % 13)
            i = 0
            while i < len(cards) - 2:
                run = [cards[i]]
                j = i + 1
                while j < len(cards) and cards[j] % 13 == (run[-1] % 13) + 1:
                    run.append(cards[j])
                    j += 1
                if len(run) >= 3:
                    melds.append(run)
                i = j if j > i + 1 else i + 1
        
        if not melds:
            return sum(min(10, x % 13 + 1) for x in hand)
        
        min_deadwood = float('inf')
        
        for i in range(1 << len(melds)):
            used = set()
            valid = True
            
            for j in range(len(melds)):
                if i & (1 << j):
                    meld = melds[j]
                    if any(card in used for card in meld):
                        valid = False
                        break
                    used.update(meld)
            
            if valid:
                unmatched = [card for card in hand if card not in used]
                deadwood = sum(min(10, x % 13 + 1) for x in unmatched)
                min_deadwood = min(min_deadwood, deadwood)
        
        return min_deadwood

    def _is_valid(self, action):
        return self._get_state()['valid_actions_mask'][action]

    def print_state(self):
        """Print the current game state in a human-readable format."""
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        def card_to_str(card_idx):
            suit = card_idx // 13
            rank = card_idx % 13
            return f"{ranks[rank]}{suits[suit]}"
        
        print("\n=== Current Game State ===")
        print(f"Current player: {self.current}")
        print(f"Phase: {self.phase}")
        
        # Print player hands
        for player in range(2):
            hand = sorted(self.hands[player], key=lambda x: (x // 13, x % 13))
            hand_str = ' '.join(card_to_str(card) for card in hand)
            if player == self.current:
                print(f"Player {player} hand (current): {hand_str}")
            else:
                print(f"Player {player} hand: {hand_str}")
        
        # Print discard pile
        if self.discard:
            top_discard = self.discard[-1]
            print(f"Top discard: {card_to_str(top_discard)}")
        else:
            print("Discard pile: empty")
        
        # Print remaining cards in deck
        print(f"Cards in deck: {len(self.deck)}")
        
        # Print deadwood for current player
        deadwood = self._get_deadwood(self.hands[self.current])
        print(f"Current player deadwood: {deadwood}")
        
        # Print valid actions
        valid_actions = []
        if self.phase == 'draw':
            valid_actions.append(f"Draw from stock ({DRAW_STOCK})")
            if self.discard:
                valid_actions.append(f"Draw from discard ({DRAW_DISCARD})")
        else:
            for card in self.hands[self.current]:
                valid_actions.append(f"Discard {card_to_str(card)} ({DISCARD_START + card})")
            
            if deadwood <= 10:
                valid_actions.append(f"Knock ({KNOCK})")
            if deadwood == 0:
                valid_actions.append(f"Gin ({GIN})")
        
        print("Valid actions:")
        for action in valid_actions:
            print(f"  {action}")
        
        print("=========================") 