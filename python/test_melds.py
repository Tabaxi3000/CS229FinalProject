#!/usr/bin/env python3

from improved_gin_rummy_env import ImprovedGinRummyEnv
import unittest
from knock import find_melds, get_deadwood

def test_meld_finding():
    env = ImprovedGinRummyEnv()
    
    # Test case 1: A set of Aces
    hand1 = [0, 13, 26, 39]  # A♠, A♥, A♦, A♣
    print("Test case 1: Set of Aces")
    print("Hand:", env._format_cards(hand1))
    melds1 = env._find_melds(hand1)
    print("Melds found:", [env._format_cards(meld) for meld in melds1])
    print("Deadwood count:", env._calculate_deadwood(hand1))
    print()
    
    # Test case 2: A run in spades
    hand2 = [0, 1, 2, 3]  # A♠, 2♠, 3♠, 4♠
    print("Test case 2: Run in spades")
    print("Hand:", env._format_cards(hand2))
    melds2 = env._find_melds(hand2)
    print("Melds found:", [env._format_cards(meld) for meld in melds2])
    print("Deadwood count:", env._calculate_deadwood(hand2))
    print()
    
    # Test case 3: A hand with both a set and a run
    hand3 = [0, 13, 26, 1, 2, 3]  # A♠, A♥, A♦, 2♠, 3♠, 4♠
    print("Test case 3: Hand with both a set and a run")
    print("Hand:", env._format_cards(hand3))
    melds3 = env._find_melds(hand3)
    print("Melds found:", [env._format_cards(meld) for meld in melds3])
    print("Deadwood count:", env._calculate_deadwood(hand3))
    print()
    
    # Test case 4: A hand with overlapping melds
    hand4 = [0, 13, 26, 1, 2, 3, 14, 27]  # A♠, A♥, A♦, 2♠, 3♠, 4♠, 2♥, 2♦
    print("Test case 4: Hand with overlapping melds")
    print("Hand:", env._format_cards(hand4))
    melds4 = env._find_melds(hand4)
    print("Melds found:", [env._format_cards(meld) for meld in melds4])
    print("Deadwood count:", env._calculate_deadwood(hand4))
    print()
    
    # Test case 5: A hand with no melds
    hand5 = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]  # Random cards
    print("Test case 5: Hand with no melds")
    print("Hand:", env._format_cards(hand5))
    melds5 = env._find_melds(hand5)
    print("Melds found:", [env._format_cards(meld) for meld in melds5])
    print("Deadwood count:", env._calculate_deadwood(hand5))
    print()
    
    # Test case 6: A hand with a gin (no deadwood)
    hand6 = [0, 1, 2, 13, 14, 15, 26, 27, 28, 39]  # A♠-2♠-3♠, A♥-2♥-3♥, A♦-2♦-3♦, A♣
    print("Test case 6: Hand with a gin (no deadwood)")
    print("Hand:", env._format_cards(hand6))
    melds6 = env._find_melds(hand6)
    print("Melds found:", [env._format_cards(meld) for meld in melds6])
    print("Deadwood count:", env._calculate_deadwood(hand6))
    print()

class TestMelds(unittest.TestCase):
    def test_empty_hand(self):
        self.assertEqual(find_melds([]), [])
        self.assertEqual(get_deadwood([]), float('inf'))
        
    def test_no_melds(self):
        hand = [0, 13, 26, 39]  # All aces
        self.assertEqual(find_melds(hand), [])
        self.assertEqual(get_deadwood(hand), 4)
        
    def test_set_meld(self):
        hand = [0, 13, 26]  # Three aces
        melds = find_melds(hand)
        self.assertEqual(len(melds), 1)
        self.assertEqual(sorted(melds[0]), sorted(hand))
        self.assertEqual(get_deadwood(hand), 0)
        
    def test_run_meld(self):
        hand = [0, 1, 2]  # A-2-3 of spades
        melds = find_melds(hand)
        self.assertEqual(len(melds), 1)
        self.assertEqual(sorted(melds[0]), sorted(hand))
        self.assertEqual(get_deadwood(hand), 0)
        
    def test_multiple_melds(self):
        hand = [0, 13, 26, 1, 2, 3]  # Three aces and 2-3-4 of spades
        melds = find_melds(hand)
        self.assertEqual(len(melds), 2)
        self.assertEqual(get_deadwood(hand), 0)
        
    def test_overlapping_melds(self):
        hand = [0, 1, 2, 13, 26]  # A-2-3 of spades and three aces
        self.assertEqual(get_deadwood(hand), 0)
        
    def test_face_cards(self):
        hand = [10, 11, 12]  # J-Q-K of spades
        self.assertEqual(get_deadwood(hand), 30)
        melds = find_melds(hand)
        self.assertEqual(len(melds), 1)
        self.assertEqual(get_deadwood(hand), 0)
        
    def test_mixed_values(self):
        hand = [0, 5, 10]  # A-6-J of spades
        self.assertEqual(get_deadwood(hand), 17)  # 1 + 6 + 10
        
if __name__ == '__main__':
    unittest.main() 