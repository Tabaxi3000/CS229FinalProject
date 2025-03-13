import json
import os

total_games = 0
game_ids = set()

# Check both potential locations for training data
locations = [
    'data/training_data_',  # Python directory
    '../java/MavenProject/training_data_'  # Java project directory
]

for location in locations:
    print(f"\nChecking location: {location}")
    
    for i in range(1, 11):
        file_path = f'{location}{i}.json'
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    print(f"Reading file {file_path}...")
                    data = json.load(f)
                    if isinstance(data, list):
                        states = data
                    else:
                        states = data.get('gameStates', [])
                    
                    # Extract all game IDs from this file
                    file_game_ids = set()
                    for state in states:
                        game_id = state.get('gameId', 0)
                        file_game_ids.add(game_id)
                    
                    print(f'File {i}: {len(file_game_ids)} unique games')
                    total_games += len(file_game_ids)
                    game_ids.update(file_game_ids)
                    
                    # Print a sample state to understand the structure
                    if states:
                        print(f'Sample state structure: {list(states[0].keys())}')
            except Exception as e:
                print(f'Error processing {file_path}: {e}')

print(f'\nTotal unique games across all files: {len(game_ids)}')
print(f'Total games (possibly with duplicates): {total_games}')
if game_ids:
    print(f'Max game ID: {max(game_ids)}')
    print(f'Min game ID: {min(game_ids)}')
    print(f'Games needed to reach 100,000: {100000 - len(game_ids)}') 