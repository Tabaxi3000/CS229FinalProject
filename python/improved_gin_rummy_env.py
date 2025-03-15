import random
import torch
import numpy as np
DRAW_STOCK = 0
DRAW_DISCARD = 1
DISCARD_START = 2
DISCARD_END = 53
KNOCK = 108
GIN = 109
class ImprovedGinRummyEnv:
    """An improved Gin Rummy environment with reward shaping for better learning."""
    def __init__(self, reward_shaping=True, verbose=False):
        self.current_player = 0
        self.deck = list(range(52))
        self.discard_pile = []
        self.player_hands = [[], []]
        self.phase = 'draw'  # 'draw' or 'discard'
        self.reward_shaping = reward_shaping
        self.verbose = verbose
        self.gin_opportunities = 0
        self.knock_opportunities = 0
        self.gin_taken = 0
        self.knock_taken = 0
        if self.verbose:
            print(f"Environment constants - GIN: {GIN}, KNOCK: {KNOCK}")
            print("Using standard Gin Rummy scoring rules")
    def reset(self):
        """Reset the environment to start a new game."""
        self.deck = list(range(52))
        random.shuffle(self.deck)
        self.player_hands = [[], []]
        for _ in range(10):
            self.player_hands[0].append(self.deck.pop())
            self.player_hands[1].append(self.deck.pop())
        self.discard_pile = [self.deck.pop()]
        self.current_player = 0
        self.phase = 'draw'
        return self._get_state()
    def step(self, action):
        """Take an action in the environment."""
        reward = 0
        done = False
        info = {
            'outcome': None,
            'player_deadwood': None,
            'opponent_deadwood': None
        }
        if action == DRAW_STOCK:
            if self.phase == 'draw':
                if self.deck:
                    card = self.deck.pop()
                    self.player_hands[self.current_player].append(card)
                    self.phase = 'discard'
                else:
                    done = True
            else:
                reward = -1
                done = True
        elif action == DRAW_DISCARD:
            if self.phase == 'draw' and self.discard_pile:
                card = self.discard_pile.pop()
                self.player_hands[self.current_player].append(card)
                self.phase = 'discard'
            else:
                reward = -1
                done = True
        elif DISCARD_START <= action <= DISCARD_END:
            card_idx = action - DISCARD_START
            if self.phase == 'discard' and card_idx in self.player_hands[self.current_player]:
                self.player_hands[self.current_player].remove(card_idx)
                self.discard_pile.append(card_idx)
                self.phase = 'draw'
                self.current_player = 1 - self.current_player
            else:
                reward = -1
                done = True
        elif action == KNOCK:
            if self.phase == 'discard':
                deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
                if self.verbose:
                    print(f"KNOCK attempt with deadwood: {deadwood}")
                if deadwood <= 10:
                    opponent_deadwood = self._calculate_deadwood(self.player_hands[1 - self.current_player])
                    if self.verbose:
                        print(f"KNOCK: Player deadwood: {deadwood}, Opponent deadwood: {opponent_deadwood}")
                    info['player_deadwood'] = deadwood
                    info['opponent_deadwood'] = opponent_deadwood
                    if deadwood < opponent_deadwood:
                        if self.verbose:
                            print(f"KNOCK WIN: Player {self.current_player} wins with {deadwood} vs {opponent_deadwood}")
                        reward = 25 + (opponent_deadwood - deadwood)
                        info['outcome'] = 'win'
                    elif deadwood > opponent_deadwood:
                        if self.verbose:
                            print(f"KNOCK LOSS: Player {self.current_player} loses with {deadwood} vs {opponent_deadwood}")
                        reward = -25 - (opponent_deadwood - deadwood)
                        info['outcome'] = 'loss'
                    else:
                        if self.verbose:
                            print(f"KNOCK DRAW: Player {self.current_player} draws with {deadwood}")
                        reward = 0
                        info['outcome'] = 'draw'
                    done = True
                else:
                    if self.verbose:
                        print(f"INVALID KNOCK: Player {self.current_player} has {deadwood} deadwood (> 10)")
                    reward = -1
                    done = True
                    info['outcome'] = 'invalid_knock'
                    info['player_deadwood'] = deadwood
            else:
                if self.verbose:
                    print(f"INVALID KNOCK: Not in discard phase")
                reward = -1
                done = True
                info['outcome'] = 'invalid_action'
        elif action == GIN:
            if self.phase == 'discard':
                deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
                if self.verbose:
                    print(f"GIN attempt with deadwood: {deadwood}")
                if deadwood == 0:
                    if self.verbose:
                        print(f"VALID GIN: Player {self.current_player} wins with gin!")
                    opponent_deadwood = self._calculate_deadwood(self.player_hands[1 - self.current_player])
                    reward = 25 + opponent_deadwood
                    info['outcome'] = 'gin'
                    info['player_deadwood'] = 0
                    info['opponent_deadwood'] = opponent_deadwood
                    done = True
                else:
                    if self.verbose:
                        print(f"INVALID GIN: Player {self.current_player} has {deadwood} deadwood (> 0)")
                    reward = -1
                    done = True
                    info['outcome'] = 'invalid_gin'
                    info['player_deadwood'] = deadwood
            else:
                if self.verbose:
                    print(f"INVALID GIN: Not in discard phase")
                reward = -1
                done = True
                info['outcome'] = 'invalid_action'
        return self._get_state(), reward, done, info
    def _get_state(self):
        """Get the current state representation."""
        hand_matrix = torch.zeros(1, 4, 13)
        for card in self.player_hands[self.current_player]:
            suit = card // 13
            rank = card % 13
            hand_matrix[0, suit, rank] = 1
        discard_history = torch.zeros(1, 10, 52)  # [batch_size, seq_len, card_idx]
        for i, card in enumerate(self.discard_pile[-10:]):
            discard_history[0, i, card] = 1
        valid_actions_mask = torch.zeros(110)
        if self.phase == 'draw':
            valid_actions_mask[DRAW_STOCK] = 1
            if self.discard_pile:
                valid_actions_mask[DRAW_DISCARD] = 1
        else:
            for card in self.player_hands[self.current_player]:
                valid_actions_mask[DISCARD_START + card] = 1
            deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
            if deadwood <= 10:
                valid_actions_mask[KNOCK] = 1
                self.knock_opportunities += 1
            if deadwood == 0:
                valid_actions_mask[GIN] = 1
                self.gin_opportunities += 1
        opponent_model = torch.zeros(52)
        for card in self.player_hands[1 - self.current_player]:
            opponent_model[card] = 1
        deadwood_count = self._calculate_deadwood(self.player_hands[self.current_player])
        return {
            'hand_matrix': hand_matrix,
            'discard_history': discard_history,
            'valid_actions_mask': valid_actions_mask,
            'current_player': self.current_player,
            'phase': self.phase,
            'opponent_model': opponent_model,
            'deadwood': deadwood_count
        }
    def _format_cards(self, cards):
        """Format cards for display."""
        suits = ['♠', '♥', '♦', '♣']
        ranks = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        return [f"{ranks[card % 13]}{suits[card // 13]}" for card in cards]
    def _calculate_deadwood(self, hand):
        """Calculate deadwood points for a hand."""
        if not hand:
            return 0
        melds = self._find_melds(hand)
        deadwood_cards = set(hand)
        for meld in melds:
            for card in meld:
                if card in deadwood_cards:
                    deadwood_cards.remove(card)
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
        ranks = [[] for _ in range(13)]
        for card in hand:
            rank = card % 13
            ranks[rank].append(card)
        suits = [[] for _ in range(4)]
        for card in hand:
            suit = card // 13
            suits[suit].append(card)
        sets = []
        for rank_cards in ranks:
            if len(rank_cards) >= 3:
                sets.append(sorted(rank_cards))
        runs = []
        for suit_cards in suits:
            if len(suit_cards) < 3:
                continue
            suit_cards.sort(key=lambda x: x % 13)
            i = 0
            while i < len(suit_cards) - 2:  # Need at least 3 cards for a run
                run = [suit_cards[i]]
                for j in range(i + 1, len(suit_cards)):
                    if suit_cards[j] % 13 == (run[-1] % 13) + 1:
                        run.append(suit_cards[j])
                    elif suit_cards[j] % 13 > (run[-1] % 13) + 1:
                        break
                if len(run) >= 3:
                    runs.append(run)
                    i += len(run) - 1  # Skip to the end of this run
                else:
                    i += 1
        all_melds = sets + runs
        def find_best_melds(remaining_cards, current_melds=None, used_cards=None):
            if current_melds is None:
                current_melds = []
            if used_cards is None:
                used_cards = set()
            if not remaining_cards:
                return current_melds, used_cards
            best_melds = current_melds.copy()
            best_used = used_cards.copy()
            for meld in all_melds:
                if all(card in remaining_cards for card in meld):
                    if not any(card in used_cards for card in meld):
                        new_remaining = [card for card in remaining_cards if card not in meld]
                        new_melds = current_melds + [meld]
                        new_used = used_cards.union(set(meld))
                        candidate_melds, candidate_used = find_best_melds(new_remaining, new_melds, new_used)
                        if len(candidate_used) > len(best_used):
                            best_melds = candidate_melds
                            best_used = candidate_used
            return best_melds, best_used
        best_melds, _ = find_best_melds(hand)
        if not best_melds:
            all_melds.sort(key=len, reverse=True)
            best_melds = []
            covered_cards = set()
            for meld in all_melds:
                new_cards = set(meld) - covered_cards
                if new_cards and len(new_cards) >= 2:  # Only add if it covers at least 2 new cards
                    best_melds.append(meld)
                    covered_cards.update(meld)
        return best_melds
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
        for player in range(2):
            hand = sorted(self.player_hands[player], key=lambda x: (x // 13, x % 13))
            hand_str = ' '.join(card_to_str(card) for card in hand)
            if player == self.current_player:
                print(f"Player {player} hand (current): {hand_str}")
            else:
                print(f"Player {player} hand: {hand_str}")
        if self.discard_pile:
            top_discard = self.discard_pile[-1]
            print(f"Top discard: {card_to_str(top_discard)}")
        else:
            print("Discard pile: empty")
        print(f"Cards in deck: {len(self.deck)}")
        deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
        print(f"Current player deadwood: {deadwood}")
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
