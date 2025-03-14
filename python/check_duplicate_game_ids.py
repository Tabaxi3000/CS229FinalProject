import json
import os
from collections import defaultdict, Counter

def check_duplicates():
    """Check for duplicate game IDs across all training data files."""
    # Directory where the JSON files are located
    json_dir = "../java/MavenProject/"
    
    # Dictionary to track which files contain each game ID
    game_id_to_files = defaultdict(list)
    
    # Counter for duplicate game IDs
    duplicate_count = 0
    
    # Check all possible training data files
    for i in range(1, 20):  # Check files 1-20
        file_path = f"{json_dir}training_data_{i}.json"
        
        if not os.path.exists(file_path):
            continue
            
        print(f"Checking file {file_path}...")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Extract game states
                if isinstance(data, list):
                    states = data
                else:
                    states = data.get('gameStates', [])
                
                # Extract all game IDs from this file and count occurrences
                file_game_ids = [state.get('gameId', 0) for state in states]
                game_id_counter = Counter(file_game_ids)
                
                # Check for duplicates within this file
                duplicates_in_file = {game_id: count for game_id, count in game_id_counter.items() if count > 1}
                if duplicates_in_file:
                    print(f"  Warning: File {i} contains duplicate game IDs internally:")
                    for game_id, count in duplicates_in_file.items():
                        print(f"    Game ID {game_id} appears {count} times")
                
                # Track which files contain each unique game ID
                for game_id in set(file_game_ids):
                    game_id_to_files[game_id].append(i)
                
                print(f"  Processed {len(states)} states with {len(set(file_game_ids))} unique game IDs")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Check for game IDs that appear in multiple files
    duplicate_game_ids = {game_id: files for game_id, files in game_id_to_files.items() if len(files) > 1}
    
    print("\n=== Summary ===")
    print(f"Total unique game IDs found: {len(game_id_to_files)}")
    print(f"Game IDs that appear in multiple files: {len(duplicate_game_ids)}")
    
    if duplicate_game_ids:
        print("\nDetailed duplicate game IDs:")
        for game_id, files in sorted(duplicate_game_ids.items()):
            print(f"Game ID {game_id} appears in files: {', '.join([str(f) for f in files])}")
    
    # Calculate ID ranges to ensure continuous coverage
    if game_id_to_files:
        min_id = min(game_id_to_files.keys())
        max_id = max(game_id_to_files.keys())
        expected_ids = set(range(min_id, max_id + 1))
        missing_ids = expected_ids - set(game_id_to_files.keys())
        
        print(f"\nID range: {min_id} to {max_id}")
        print(f"Missing IDs in range: {len(missing_ids)}")
        if len(missing_ids) < 20:  # Only show if there are few missing IDs
            print(f"Missing IDs: {sorted(missing_ids)}")
    
    return len(duplicate_game_ids) == 0

if __name__ == "__main__":
    no_duplicates = check_duplicates()
    if no_duplicates:
        print("\nNo duplicate game IDs found across files! The data is clean.")
    else:
        print("\nDuplicate game IDs found! This may cause issues when training the model.") 