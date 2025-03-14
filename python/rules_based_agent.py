import numpy as np
from typing import List, Dict, Tuple, Optional
from simple_evaluate import DRAW_STOCK, DRAW_DISCARD, DISCARD, KNOCK, GIN

class RulesBasedAgent:
    """A rules-based agent that implements basic Gin Rummy strategy."""
    
    def __init__(self):
        self.deadwood_threshold = 10  # Threshold for knocking
        
    def select_action(self, state: Dict) -> int:
        """Select an action based on hand state and basic strategy rules."""
        hand_matrix = state['hand_matrix']
        discard_pile = state['discard_history']
        valid_actions = state['valid_actions_mask']
        
        # Convert matrices to card lists for easier processing
        hand = self._matrix_to_cards(hand_matrix[0])  # Remove batch dimension
        deadwood, melds = self.calculate_deadwood_and_melds(hand)
        
        # Check if we can knock or gin
        if deadwood <= self.deadwood_threshold and valid_actions[KNOCK]:
            return KNOCK
        
        # Analyze discard pile top card
        top_discard = self._get_top_discard(discard_pile)
        if top_discard is not None:
            # Check if taking discard would improve hand
            potential_hand = hand + [top_discard]
            new_deadwood, new_melds = self.calculate_deadwood_and_melds(potential_hand)
            
            if new_deadwood < deadwood and valid_actions[DRAW_DISCARD]:
                return DRAW_DISCARD
        
        # If we can't improve with discard, draw from stock
        if valid_actions[DRAW_STOCK]:
            return DRAW_STOCK
        
        # If we must discard, choose highest deadwood card
        if any(valid_actions[52:104]):  # Discard actions
            worst_card = self._find_worst_card(hand, melds)
            return 52 + worst_card  # Offset for discard action
            
        return DRAW_STOCK  # Default action
    
    def _matrix_to_cards(self, matrix: np.ndarray) -> List[int]:
        """Convert a hand matrix to list of card indices."""
        cards = []
        for suit in range(4):
            for rank in range(13):
                if matrix[suit, rank] == 1:
                    cards.append(suit * 13 + rank)
        return cards
    
    def _get_top_discard(self, discard_history: np.ndarray) -> Optional[int]:
        """Get the top card from discard pile."""
        if discard_history.shape[1] == 0:
            return None
        
        # Find last non-zero entry
        for i in range(discard_history.shape[1] - 1, -1, -1):
            if (discard_history[0, i] != 0).any():
                rank = np.argmax(discard_history[0, i])
                # Assume it's from first suit (this is a simplification)
                return rank
        return None
    
    def _find_worst_card(self, hand: List[int], melds: List[List[int]]) -> int:
        """Find the card contributing most to deadwood."""
        if not hand:
            return 0
            
        # Get unmelded cards
        melded = set()
        for meld in melds:
            melded.update(meld)
        unmelded = [card for card in hand if card not in melded]
        
        if not unmelded:
            # If all cards are melded, remove from longest meld
            longest_meld = max(melds, key=len)
            return longest_meld[-1]
        
        # Return highest point card from unmelded
        return max(unmelded, key=lambda x: min(10, (x % 13) + 1))
    
    def calculate_deadwood_and_melds(self, hand: List[int]) -> Tuple[int, List[List[int]]]:
        """Calculate deadwood points and find melds in a hand."""
        if not hand:
            return float('inf'), []
        
        # Sort hand by rank within each suit
        sorted_hand = sorted(hand)
        
        # Find all possible melds
        melds = []
        
        # Find sets (same rank, different suits)
        rank_groups = {}
        for card in sorted_hand:
            rank = card % 13
            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(card)
        
        for rank, cards in rank_groups.items():
            if len(cards) >= 3:
                melds.append(cards[:])
        
        # Find runs (consecutive ranks, same suit)
        suit_groups = {}
        for card in sorted_hand:
            suit = card // 13
            if suit not in suit_groups:
                suit_groups[suit] = []
            suit_groups[suit].append(card)
        
        for suit, cards in suit_groups.items():
            cards.sort(key=lambda x: x % 13)
            i = 0
            while i < len(cards):
                run = [cards[i]]
                j = i + 1
                while j < len(cards) and cards[j] % 13 == (cards[j-1] % 13) + 1:
                    run.append(cards[j])
                    j += 1
                if len(run) >= 3:
                    melds.append(run)
                i = j
        
        # Calculate minimum deadwood
        if not melds:
            return sum(min(10, x % 13 + 1) for x in hand), []
        
        # Find best combination of non-overlapping melds
        min_deadwood = float('inf')
        best_melds = []
        
        def get_deadwood(used_cards):
            return sum(min(10, x % 13 + 1) for x in hand if x not in used_cards)
        
        def try_melds(used_melds, used_cards, meld_idx):
            nonlocal min_deadwood, best_melds
            
            # Calculate current deadwood
            current_deadwood = get_deadwood(used_cards)
            if current_deadwood < min_deadwood:
                min_deadwood = current_deadwood
                best_melds = used_melds[:]
            
            # Try adding each remaining meld
            for i in range(meld_idx, len(melds)):
                meld = melds[i]
                if not any(card in used_cards for card in meld):
                    used_melds.append(meld)
                    used_cards.update(meld)
                    try_melds(used_melds, used_cards, i + 1)
                    used_melds.pop()
                    for card in meld:
                        used_cards.remove(card)
        
        try_melds([], set(), 0)
        return min_deadwood, best_melds 