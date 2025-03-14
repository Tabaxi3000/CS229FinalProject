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

class ImprovedGinRummyEnv:
    """An improved Gin Rummy environment with reward shaping for better learning."""
    
    def __init__(self, reward_shaping=True, deadwood_reward_scale=0.01, meld_reward=0.05, win_reward=1.0, gin_reward=1.5, knock_reward=0.5):
        self.current_player = 0
        self.deck = list(range(52))
        self.discard_pile = []
        self.player_hands = [[], []]
        self.phase = 'draw'  # 'draw' or 'discard'
        self.reward_shaping = reward_shaping
        self.deadwood_reward_scale = deadwood_reward_scale
        self.meld_reward = meld_reward
        self.win_reward = win_reward
        self.gin_reward = gin_reward
        self.knock_reward = knock_reward  # Added explicit knock reward
        self.previous_deadwood = [100, 100]  # Initialize with high values
        self.previous_melds_count = [0, 0]  # Track number of melds
        
        # Debug counters
        self.gin_opportunities = 0
        self.knock_opportunities = 0
        self.gin_taken = 0
        self.knock_taken = 0
        
        # Print constants for debugging
        print(f"Environment constants - GIN: {GIN}, KNOCK: {KNOCK}")
        print(f"Reward structure - win: {self.win_reward}, gin: {self.gin_reward}, knock: {self.knock_reward}")
        
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
        
        # Reset tracking variables for reward shaping
        self.previous_deadwood = [
            self._calculate_deadwood(self.player_hands[0]),
            self._calculate_deadwood(self.player_hands[1])
        ]
        self.previous_melds_count = [
            len(self._find_melds(self.player_hands[0])),
            len(self._find_melds(self.player_hands[1]))
        ]
        
        # Create state representation
        return self._get_state()
    
    def step(self, action):
        """Take an action in the environment."""
        reward = 0
        shaped_reward = 0
        done = False
        truncated = False  # Add truncated flag for gym compatibility
        info = {
            'original_reward': 0,
            'outcome': None,
            'player_deadwood': None,
            'opponent_deadwood': None
        }
        
        # Store previous state for reward shaping
        prev_deadwood = self.previous_deadwood[self.current_player]
        prev_melds_count = self.previous_melds_count[self.current_player]
        
        # Debug action
        if action == GIN:
            print(f"GIN action taken by player {self.current_player}")
            self.gin_taken += 1
        elif action == KNOCK:
            print(f"KNOCK action taken by player {self.current_player}")
            self.knock_taken += 1
        
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
                shaped_reward = -1
                done = True
                
        elif action == DRAW_DISCARD:
            # Draw from discard
            if self.phase == 'draw' and self.discard_pile:
                card = self.discard_pile.pop()
                self.player_hands[self.current_player].append(card)
                self.phase = 'discard'
                
                # Small reward for drawing a card that could form a meld
                if self.reward_shaping:
                    temp_hand = self.player_hands[self.current_player].copy()
                    temp_hand.remove(card)  # Remove the card to check if it helps form a new meld
                    old_melds = self._find_melds(temp_hand)
                    new_melds = self._find_melds(self.player_hands[self.current_player])
                    if len(new_melds) > len(old_melds):
                        shaped_reward += self.meld_reward
            else:
                # Invalid action
                reward = -1
                shaped_reward = -1
                done = True
                
        elif DISCARD_START <= action <= DISCARD_END:
            # Discard a card
            card_idx = action - DISCARD_START
            if self.phase == 'discard' and card_idx in self.player_hands[self.current_player]:
                # Calculate deadwood before discard
                current_deadwood_before = self._calculate_deadwood(self.player_hands[self.current_player])
                
                # Remove card from hand
                self.player_hands[self.current_player].remove(card_idx)
                self.discard_pile.append(card_idx)
                
                # Calculate deadwood after discard
                current_deadwood_after = self._calculate_deadwood(self.player_hands[self.current_player])
                
                # Reward for reducing deadwood
                if self.reward_shaping:
                    deadwood_reduction = current_deadwood_before - current_deadwood_after
                    if deadwood_reduction > 0:
                        shaped_reward += deadwood_reduction * self.deadwood_reward_scale
                
                self.phase = 'draw'
                # Switch player
                self.current_player = 1 - self.current_player
            else:
                # Invalid action
                reward = -1
                shaped_reward = -1
                done = True
                
        elif action == KNOCK:
            # Knock
            if self.phase == 'discard':
                deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
                print(f"KNOCK attempt with deadwood: {deadwood}")
                if deadwood <= 10:
                    # Valid knock
                    opponent_deadwood = self._calculate_deadwood(self.player_hands[1 - self.current_player])
                    print(f"KNOCK: Player deadwood: {deadwood}, Opponent deadwood: {opponent_deadwood}")
                    
                    # Add explicit knock reward regardless of outcome
                    shaped_reward += self.knock_reward
                    
                    # Set info dictionary values
                    info['player_deadwood'] = deadwood
                    info['opponent_deadwood'] = opponent_deadwood
                    
                    if deadwood < opponent_deadwood:
                        print(f"KNOCK WIN: Player {self.current_player} wins with {deadwood} vs {opponent_deadwood}")
                        # CRITICAL FIX: Always give positive reward for winning
                        reward = self.win_reward  # Win with configurable reward
                        shaped_reward += self.win_reward
                        # Set outcome in info dictionary
                        info['outcome'] = 'win'
                        info['original_reward'] = self.win_reward
                        # Bonus for winning with low deadwood
                        if self.reward_shaping:
                            shaped_reward += (10 - deadwood) * 0.05
                    elif deadwood > opponent_deadwood:
                        print(f"KNOCK LOSS: Player {self.current_player} loses with {deadwood} vs {opponent_deadwood}")
                        reward = -self.win_reward  # Loss (negative win reward)
                        shaped_reward = -self.win_reward
                        # Set outcome in info dictionary
                        info['outcome'] = 'loss'
                        info['original_reward'] = -self.win_reward
                    else:
                        print(f"KNOCK DRAW: Player {self.current_player} draws with {deadwood}")
                        reward = 0  # Draw
                        shaped_reward = 0
                        # Set outcome in info dictionary
                        info['outcome'] = 'draw'
                        info['original_reward'] = 0
                    
                    # CRITICAL FIX: Always end the game after KNOCK
                    done = True
                else:
                    # Invalid knock (deadwood > 10)
                    print(f"INVALID KNOCK: Player {self.current_player} has {deadwood} deadwood (> 10)")
                    reward = -1
                    shaped_reward = -1
                    done = True
                    # Set outcome in info dictionary
                    info['outcome'] = 'invalid_knock'
                    info['original_reward'] = -1
                    info['player_deadwood'] = deadwood
            else:
                # Invalid action
                print(f"INVALID KNOCK: Not in discard phase")
                reward = -1
                shaped_reward = -1
                done = True
                # Set outcome in info dictionary
                info['outcome'] = 'invalid_action'
                info['original_reward'] = -1
                
        elif action == GIN:
            # Gin
            if self.phase == 'discard':
                deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
                print(f"GIN attempt with deadwood: {deadwood}")
                if deadwood == 0:
                    # Valid gin
                    print(f"VALID GIN: Player {self.current_player} wins with gin!")
                    # CRITICAL FIX: Always give positive reward for winning with gin
                    reward = self.gin_reward  # Gin with configurable reward
                    shaped_reward = self.gin_reward  # Extra reward for gin
                    
                    # Set outcome in info dictionary
                    info['outcome'] = 'gin'
                    info['original_reward'] = self.gin_reward
                    info['player_deadwood'] = 0
                    info['opponent_deadwood'] = self._calculate_deadwood(self.player_hands[1 - self.current_player])
                    
                    # CRITICAL FIX: Always end the game after GIN
                    done = True
                else:
                    # Invalid gin (deadwood > 0)
                    print(f"INVALID GIN: Player {self.current_player} has {deadwood} deadwood (> 0)")
                    reward = -1
                    shaped_reward = -1
                    done = True
                    
                    # Set outcome in info dictionary
                    info['outcome'] = 'invalid_gin'
                    info['original_reward'] = -1
                    info['player_deadwood'] = deadwood
            else:
                # Invalid action
                print(f"INVALID GIN: Not in discard phase")
                reward = -1
                shaped_reward = -1
                done = True
                
                # Set outcome in info dictionary
                info['outcome'] = 'invalid_action'
                info['original_reward'] = -1
        
        # Update tracking variables for next step
        if not done and self.current_player == 0:  # Only track for the learning agent
            current_deadwood = self._calculate_deadwood(self.player_hands[self.current_player])
            current_melds_count = len(self._find_melds(self.player_hands[self.current_player]))
            
            # Update previous values
            self.previous_deadwood[self.current_player] = current_deadwood
            self.previous_melds_count[self.current_player] = current_melds_count
        
        # Use shaped reward if reward shaping is enabled, otherwise use original reward
        final_reward = shaped_reward if self.reward_shaping else reward
        
        # Store original reward in info
        info['original_reward'] = reward
        info['shaped_reward'] = shaped_reward
        
        # Return next state, reward, done, truncated, info (5 values instead of 4)
        return self._get_state(), final_reward, done, truncated, info
    
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
            
            # Print detailed information about the hand and melds
            if self.current_player == 0:  # Only for the learning agent
                melds = self._find_melds(self.player_hands[self.current_player])
                deadwood_cards = set(self.player_hands[self.current_player])
                for meld in melds:
                    for card in meld:
                        if card in deadwood_cards:
                            deadwood_cards.remove(card)
                
                print(f"\nPlayer {self.current_player} hand: {self._format_cards(self.player_hands[self.current_player])}")
                print(f"Melds found: {[self._format_cards(meld) for meld in melds]}")
                print(f"Deadwood cards: {self._format_cards(list(deadwood_cards))}")
                print(f"Deadwood count: {deadwood}")
            
            if deadwood <= 10:
                valid_actions_mask[KNOCK] = 1
                self.knock_opportunities += 1
                if self.current_player == 0:  # Only log for the learning agent
                    print(f"KNOCK opportunity: Player {self.current_player} has {deadwood} deadwood")
            
            if deadwood == 0:
                valid_actions_mask[GIN] = 1
                self.gin_opportunities += 1
                if self.current_player == 0:  # Only log for the learning agent
                    print(f"GIN opportunity: Player {self.current_player} has 0 deadwood")
        
        # Create opponent model (for MCTS)
        opponent_model = torch.zeros(52)
        for card in self.player_hands[1 - self.current_player]:
            opponent_model[card] = 1
        
        # Add deadwood count to the state for better learning
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
        
        # Find sets (3 or 4 cards of the same rank)
        sets = []
        for rank_cards in ranks:
            if len(rank_cards) >= 3:
                sets.append(sorted(rank_cards))
        
        # Find runs (3 or more consecutive cards of the same suit)
        runs = []
        for suit_cards in suits:
            if len(suit_cards) < 3:
                continue
                
            # Sort by rank
            suit_cards.sort(key=lambda x: x % 13)
            
            # Find all possible runs of length 3 or more
            i = 0
            while i < len(suit_cards) - 2:  # Need at least 3 cards for a run
                run = [suit_cards[i]]
                for j in range(i + 1, len(suit_cards)):
                    # Check if this card continues the run
                    if suit_cards[j] % 13 == (run[-1] % 13) + 1:
                        run.append(suit_cards[j])
                    # If there's a gap, stop this run
                    elif suit_cards[j] % 13 > (run[-1] % 13) + 1:
                        break
                
                # If we found a valid run of 3 or more cards, add it
                if len(run) >= 3:
                    runs.append(run)
                    i += len(run) - 1  # Skip to the end of this run
                else:
                    i += 1
        
        # Combine all melds
        all_melds = sets + runs
        
        # Use a recursive approach to find the best combination of melds
        def find_best_melds(remaining_cards, current_melds=None, used_cards=None):
            if current_melds is None:
                current_melds = []
            if used_cards is None:
                used_cards = set()
            
            # Base case: no more cards to consider
            if not remaining_cards:
                return current_melds, used_cards
            
            best_melds = current_melds.copy()
            best_used = used_cards.copy()
            
            # Try each possible meld
            for meld in all_melds:
                # Check if this meld uses only remaining cards
                if all(card in remaining_cards for card in meld):
                    # Check if this meld doesn't overlap with used cards
                    if not any(card in used_cards for card in meld):
                        # Use this meld
                        new_remaining = [card for card in remaining_cards if card not in meld]
                        new_melds = current_melds + [meld]
                        new_used = used_cards.union(set(meld))
                        
                        # Recursively find the best melds for the remaining cards
                        candidate_melds, candidate_used = find_best_melds(new_remaining, new_melds, new_used)
                        
                        # If this combination uses more cards, it's better
                        if len(candidate_used) > len(best_used):
                            best_melds = candidate_melds
                            best_used = candidate_used
            
            return best_melds, best_used
        
        # Find the best combination of melds
        best_melds, _ = find_best_melds(hand)
        
        # If the recursive approach is too slow or causes issues, fall back to the greedy approach
        if not best_melds:
            # Sort melds by length (descending)
            all_melds.sort(key=len, reverse=True)
            
            best_melds = []
            covered_cards = set()
            
            for meld in all_melds:
                # Check if this meld covers new cards
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