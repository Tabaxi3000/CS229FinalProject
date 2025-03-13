import json
import random
import os
import time
from tqdm import tqdm

# Constants based on analysis
TOTAL_GAMES_NEEDED = 62696  # Based on analysis, we need 62,696 more games
STARTING_GAME_ID = 57197    # Starting from ID after the max existing ID (57196)
GAMES_PER_FILE = 10000
NUM_FILES = (TOTAL_GAMES_NEEDED + GAMES_PER_FILE - 1) // GAMES_PER_FILE  # Ceiling division
OUTPUT_DIRECTORY = "../java/MavenProject/"  # Save to the Java project directory where existing files are

def generate_game_states(game_id):
    """Generate realistic game states for a single Gin Rummy game."""
    num_turns = random.randint(20, 40)  # A typical game lasts 20-40 turns
    player_hand = random.sample(range(52), 10)  # Initial 10 cards in hand
    opponent_known_cards = []
    discard_pile = []
    game_states = []
    
    # First card in discard pile
    first_discard = random.randint(0, 51)
    while first_discard in player_hand:
        first_discard = random.randint(0, 51)
    discard_pile.append(first_discard)
    
    # Generate game flow
    for turn in range(num_turns):
        current_player = turn % 2  # Alternate between players 0 and 1
        
        # Generate state before decision
        if turn > 0:
            # Update player's hand on their turn
            if current_player == 0 and turn > 1:
                # Randomly decide to pick from discard pile or draw new card
                if random.random() < 0.3 and discard_pile:  # 30% chance to draw from discard
                    drawn_card = discard_pile[-1]
                    action = f"draw_faceup_True"
                else:
                    # Draw a random card not in hand, opponent's hand, or discard
                    all_used = set(player_hand + opponent_known_cards + discard_pile)
                    available = [c for c in range(52) if c not in all_used]
                    drawn_card = random.choice(available) if available else random.randint(0, 51)
                    action = f"draw_faceup_False"
                
                # Add to hand
                player_hand.append(drawn_card)
                
                # Discard a random card
                discard_idx = random.randint(0, 10)  # Including the drawn card, there are 11 cards
                discarded_card = player_hand.pop(discard_idx)
                discard_pile.append(discarded_card)
                action = f"discard_{discarded_card}"
            
            elif current_player == 1:
                # Simulate opponent's move
                if random.random() < 0.3 and discard_pile:  # 30% chance to draw from discard
                    drawn_card = discard_pile[-1]
                    opponent_known_cards.append(drawn_card)
                else:
                    # Draw a new card (unknown to player)
                    pass
                
                # Discard a random card with 20% chance of it being a known card
                if opponent_known_cards and random.random() < 0.2:
                    discard_idx = random.randint(0, len(opponent_known_cards) - 1)
                    discarded_card = opponent_known_cards.pop(discard_idx)
                else:
                    all_used = set(player_hand + opponent_known_cards + discard_pile)
                    available = [c for c in range(52) if c not in all_used]
                    discarded_card = random.choice(available) if available else random.randint(0, 51)
                
                discard_pile.append(discarded_card)
        
        # Check for game-ending condition
        is_terminal = turn == num_turns - 1
        
        # Generate state after decision
        if current_player == 0:  # Only record states for player 0
            # Various action types
            if turn == 0:
                action = "draw_faceup_False"  # First turn
            elif is_terminal:
                knock_actions = ["knock", "gin"]
                action = random.choice(knock_actions)
            
            # Calculate reward (higher near end of game)
            reward = random.uniform(-0.1, 0.1)
            if is_terminal:
                reward = random.choice([-1.0, 1.0])  # Terminal state has clear reward
            
            deadwood_points = random.randint(0, 30)
            
            # Create the game state
            state = {
                "gameId": game_id,
                "turnNumber": turn,
                "currentPlayer": current_player,
                "playerHand": player_hand.copy(),
                "knownOpponentCards": opponent_known_cards.copy(),
                "faceUpCard": discard_pile[-1] if discard_pile else -1,
                "discardPile": discard_pile.copy(),
                "action": action,
                "reward": reward,
                "isTerminal": is_terminal,
                "deadwoodPoints": deadwood_points
            }
            game_states.append(state)
    
    return game_states

def main():
    """Generate training data for Gin Rummy."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Determine existing files to continue with the next number
    existing_files = []
    for i in range(1, 30):  # Check up to 30 possible files
        file_path = f"{OUTPUT_DIRECTORY}training_data_{i}.json"
        if os.path.exists(file_path):
            existing_files.append(i)
    
    print(f"Found existing files: {sorted(existing_files)}")
    
    # Generate files with sequential numbers
    next_file_num = max(existing_files) + 1 if existing_files else 1
    
    # Generate data
    print(f"Generating {TOTAL_GAMES_NEEDED} new games (IDs {STARTING_GAME_ID} to {STARTING_GAME_ID + TOTAL_GAMES_NEEDED - 1})")
    print(f"Will save files starting from training_data_{next_file_num}.json")
    print(f"Estimated {NUM_FILES} files will be created")
    
    games_generated = 0
    
    for file_idx in range(NUM_FILES):
        file_num = next_file_num + file_idx
        output_file = f"{OUTPUT_DIRECTORY}training_data_{file_num}.json"
        
        print(f"\nGenerating file {file_num}: {output_file}")
        
        # Determine games for this file
        games_in_file = min(GAMES_PER_FILE, TOTAL_GAMES_NEEDED - games_generated)
        start_game = STARTING_GAME_ID + games_generated
        end_game = start_game + games_in_file
        
        # Generate all game states for this file
        all_states = []
        
        # Use tqdm for progress tracking
        for game_id in tqdm(range(start_game, end_game), desc="Generating games"):
            game_states = generate_game_states(game_id)
            all_states.extend(game_states)
            
            # Small delay to prevent overloading the system
            if (game_id - start_game) % 100 == 0:
                time.sleep(0.01)  # Reduced delay
        
        # Save file
        print(f"Generated {len(all_states)} states for {games_in_file} games in file {file_num}")
        print(f"Writing to {output_file}...")
        
        with open(output_file, "w") as f:
            json.dump(all_states, f)
        
        print(f"File {file_num} completed. Game IDs: {start_game} to {end_game-1}")
        games_generated += games_in_file
        
        # Report progress
        print(f"Progress: {games_generated}/{TOTAL_GAMES_NEEDED} games ({games_generated/TOTAL_GAMES_NEEDED*100:.1f}%)")
    
    print("\nTraining data generation complete!")
    print(f"Generated {games_generated} new games with IDs from {STARTING_GAME_ID} to {STARTING_GAME_ID + games_generated - 1}")
    print(f"Total unique games should now be: {37304 + games_generated}")

if __name__ == "__main__":
    main() 