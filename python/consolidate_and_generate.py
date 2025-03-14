import json
import random
import os
import time
from collections import defaultdict

# Constants
TOTAL_UNIQUE_GAMES_NEEDED = 100000  # We want 100,000 total unique games
GAMES_PER_FILE = 10000
OUTPUT_DIRECTORY = "../java/MavenProject/"

def load_existing_data():
    """Load and deduplicate all existing training data."""
    game_id_to_states = defaultdict(list)  # Map game IDs to their states
    existing_game_ids = set()
    
    json_dir = OUTPUT_DIRECTORY
    
    # Read all files and consolidate data
    print("Loading and deduplicating existing data...")
    for i in range(1, 20):  # Check files 1-20
        file_path = f"{json_dir}training_data_{i}.json"
        
        if not os.path.exists(file_path):
            continue
            
        print(f"Processing file {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract game states
                if isinstance(data, list):
                    states = data
                else:
                    states = data.get('gameStates', [])
                
                # Group states by game ID, prioritizing later files (overwrite earlier)
                for state in states:
                    game_id = state.get('gameId', 0)
                    existing_game_ids.add(game_id)
                    
                    # Only store if this is the first time we've seen this game
                    # or if we're in a higher-numbered file (takes precedence)
                    if game_id not in game_id_to_states:
                        game_id_to_states[game_id].append(state)
                    else:
                        # If we've already seen this game ID, we'll prioritize the later file
                        # by appending to the existing list
                        game_id_to_states[game_id].append(state)
                
                print(f"  Added {len(states)} states")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    unique_game_count = len(game_id_to_states)
    print(f"\nFound {unique_game_count} unique games.")
    print(f"Total unique game IDs: {len(existing_game_ids)}")
    
    return game_id_to_states, max(existing_game_ids) + 1 if existing_game_ids else 0

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
    """Consolidate existing data and generate new unique data."""
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    
    # Step 1: Load existing data and find how many unique games we already have
    game_id_to_states, next_game_id = load_existing_data()
    existing_unique_games = len(game_id_to_states)
    
    # Step 2: Determine how many new games we need to generate
    new_games_needed = TOTAL_UNIQUE_GAMES_NEEDED - existing_unique_games
    if new_games_needed < 0:
        new_games_needed = 0
    
    print(f"\nWill generate {new_games_needed} new games with IDs starting from {next_game_id}")
    
    # Step 3: Generate new games
    all_states = []
    total_games = new_games_needed
    for game_id in range(next_game_id, next_game_id + new_games_needed):
        # Print progress
        if game_id % 100 == 0:
            progress = (game_id - next_game_id) / total_games * 100 if total_games > 0 else 100
            print(f"Generating game {game_id-next_game_id+1}/{total_games} ({progress:.1f}%)")
        
        game_states = generate_game_states(game_id)
        all_states.extend(game_states)
        
        # Every 1000 games, dump to a file to avoid memory issues
        if len(all_states) > GAMES_PER_FILE * 30:  # ~30 states per game
            file_index = (game_id - next_game_id) // GAMES_PER_FILE + 11  # Start from file 11
            output_file = f"{OUTPUT_DIRECTORY}training_data_{file_index}.json"
            print(f"\nWriting batch to {output_file}...")
            
            with open(output_file, "w") as f:
                json.dump(all_states, f)
            
            all_states = []  # Reset for next batch
            
        # Small delay to prevent overloading the system
        if (game_id - next_game_id) % 100 == 0:
            time.sleep(0.01)
    
    # Save any remaining states
    if all_states:
        file_index = (next_game_id + new_games_needed - next_game_id) // GAMES_PER_FILE + 11
        output_file = f"{OUTPUT_DIRECTORY}training_data_{file_index}.json"
        print(f"\nWriting remaining states to {output_file}...")
        
        with open(output_file, "w") as f:
            json.dump(all_states, f)
    
    # Step 4: Now save the consolidated existing games to new files
    print("\nNow consolidating existing games to new files with unique game IDs...")
    
    # Group existing states by their game ID
    existing_states = []
    for game_id, states in game_id_to_states.items():
        existing_states.extend(states)
    
    # Split and save to files
    chunk_size = GAMES_PER_FILE * 30  # ~30 states per game
    for i in range(0, len(existing_states), chunk_size):
        chunk = existing_states[i:i + chunk_size]
        idx = i // chunk_size + 1
        output_file = f"{OUTPUT_DIRECTORY}training_data_consolidated_{idx}.json"
        print(f"Writing consolidated existing data to {output_file}...")
        
        with open(output_file, "w") as f:
            json.dump(chunk, f)
    
    print("\nConsolidation and generation complete!")
    print(f"Total unique games: {existing_unique_games + new_games_needed}")
    print("You can now use these files for training.")
    print("For training, use consolidated_*.json files for existing data, and training_data_11+.json for new data.")

if __name__ == "__main__":
    main() 