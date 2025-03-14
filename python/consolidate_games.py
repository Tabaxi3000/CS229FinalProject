import json
import os
from collections import defaultdict
import math

# Constants
OUTPUT_DIRECTORY = "../java/MavenProject/"
TARGET_FILES = 10  # We want exactly 10 consolidated files
TARGET_GAMES_PER_FILE = 10000  # Ideal number of games per file

def load_and_consolidate():
    """Load all existing training data and consolidate into 10 files."""
    print("Starting consolidation process...")
    
    # Step 1: Load all existing game states
    all_game_states = defaultdict(list)  # Map game IDs to their states
    existing_game_ids = set()
    
    # First check for existing files (both original and any previously consolidated files)
    file_patterns = [
        "training_data_",
        "training_data_consolidated_"
    ]
    
    for pattern in file_patterns:
        for i in range(1, 30):  # Check a reasonable range of files
            file_path = f"{OUTPUT_DIRECTORY}{pattern}{i}.json"
            
            if not os.path.exists(file_path):
                continue
                
            print(f"Reading file {file_path}...")
            
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                    # Extract game states
                    if isinstance(data, list):
                        states = data
                    else:
                        states = data.get('gameStates', [])
                    
                    # Group states by game ID
                    for state in states:
                        game_id = state.get('gameId', 0)
                        all_game_states[game_id].append(state)
                        existing_game_ids.add(game_id)
                    
                    print(f"  Processed {len(states)} states")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Step 2: Calculate statistics
    total_unique_games = len(all_game_states)
    total_states = sum(len(states) for states in all_game_states.values())
    
    print(f"\nStatistics:")
    print(f"Total unique games found: {total_unique_games}")
    print(f"Total states: {total_states}")
    print(f"Average states per game: {total_states / total_unique_games:.1f}")
    
    # Step 3: Determine how to distribute the games
    games_per_file = math.ceil(total_unique_games / TARGET_FILES)
    print(f"\nDistributing {total_unique_games} games across {TARGET_FILES} files")
    print(f"Each file will contain approximately {games_per_file} games")
    
    # Step 4: Split the games into batches and save to files
    game_ids = sorted(all_game_states.keys())
    
    # Delete any existing consolidated files to avoid confusion
    for i in range(1, 30):
        file_path = f"{OUTPUT_DIRECTORY}training_data_consolidated_{i}.json"
        if os.path.exists(file_path):
            print(f"Removing existing file: {file_path}")
            os.remove(file_path)
    
    # Create new consolidated files
    for file_idx in range(TARGET_FILES):
        start_idx = file_idx * games_per_file
        end_idx = min((file_idx + 1) * games_per_file, len(game_ids))
        
        if start_idx >= len(game_ids):
            break  # No more games to process
        
        file_game_ids = game_ids[start_idx:end_idx]
        file_states = []
        
        for game_id in file_game_ids:
            file_states.extend(all_game_states[game_id])
        
        output_file = f"{OUTPUT_DIRECTORY}training_data_consolidated_{file_idx + 1}.json"
        print(f"Creating file {output_file} with {len(file_game_ids)} games ({len(file_states)} states)")
        
        with open(output_file, 'w') as f:
            json.dump(file_states, f)
    
    print("\nConsolidation complete!")
    print(f"Created {min(TARGET_FILES, math.ceil(total_unique_games / games_per_file))} consolidated files.")

if __name__ == "__main__":
    load_and_consolidate() 