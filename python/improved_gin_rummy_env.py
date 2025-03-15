import numpy as np
import random
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN

class GinRummy:
    def __init__(self, seed=None):
        self.rng = np.random.RandomState(seed)
        self.reset()
        
    def reset(self):
        self.deck = list(range(52))
        self.rng.shuffle(self.deck)
        
        self.hands = [self.deck[:10], self.deck[10:20]]
        self.hands = [sorted(hand) for hand in hand]
        
        self.discard = self.deck[20:21]
        self.stock = self.deck[21:]
        
        self.current = 0
        self.phase = 'draw'
        self.last_draw = None
        self.history = []
        
        return self._get_state()
        
    def step(self, action):
        if not self._is_valid(action):
            return self._get_state(), -50, True
            
        reward = 0
        done = False
        
        if action == DRAW_STOCK:
            if not self.stock:
                return self._get_state(), 0, True
            card = self.stock.pop(0)
            self.hands[self.current].append(card)
            self.last_draw = card
            self.phase = 'discard'
            
        elif action == DRAW_DISCARD:
            if not self.discard:
                return self._get_state(), 0, True
            card = self.discard.pop()
            self.hands[self.current].append(card)
            self.last_draw = card
            self.phase = 'discard'
            
        elif DISCARD <= action < KNOCK:
            card = action - DISCARD
            if card not in self.hands[self.current]:
                return self._get_state(), -50, True
            self.hands[self.current].remove(card)
            self.discard.append(card)
            self.history.append(card)
            self.phase = 'draw'
            self.current = 1 - self.current
            
        elif action in (KNOCK, GIN):
            deadwood = [self._get_deadwood(h) for h in self.hands]
            
            if action == GIN and deadwood[self.current] == 0:
                reward = 25
            elif action == KNOCK and deadwood[self.current] <= 10:
                if deadwood[self.current] < deadwood[1 - self.current]:
                    reward = 10
                else:
                    reward = -10
            else:
                reward = -25
                
            done = True
            
        return self._get_state(), reward, done
        
    def _get_state(self):
        hand = self._hand_matrix(self.hands[self.current])
        oppo = self._hand_matrix(self.hands[1 - self.current])
        
        discard_top = self.discard[-1] if self.discard else -1
        discard_hist = np.zeros((52, 13))
        for i, card in enumerate(self.history[-52:]):
            discard_hist[i, card % 13] = 1
            
        valid = self._valid_actions()
        
        return {
            'hand_matrix': hand,
            'opponent_matrix': oppo,
            'discard_top': discard_top,
            'discard_history': discard_hist,
            'valid_actions_mask': valid,
            'phase': self.phase,
            'current_player': self.current
        }
        
    def _hand_matrix(self, hand):
        matrix = np.zeros((4, 13))
        for card in hand:
            suit, rank = divmod(card, 13)
            matrix[suit, rank] = 1
        return matrix[None, ...]
        
    def _valid_actions(self):
        valid = np.zeros(110, dtype=bool)
        
        if self.phase == 'draw':
            if self.stock:
                valid[DRAW_STOCK] = True
            if self.discard:
                valid[DRAW_DISCARD] = True
                
        elif self.phase == 'discard':
            hand = self.hands[self.current]
            deadwood = self._get_deadwood(hand)
            
            for card in hand:
                valid[DISCARD + card] = True
                
            if deadwood == 0:
                valid[GIN] = True
            elif deadwood <= 10:
                valid[KNOCK] = True
                
        return valid
        
    def _get_deadwood(self, hand):
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
        return self._valid_actions()[action] 
